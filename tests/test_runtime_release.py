from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simready_bench import runtime_release as runtime
from simready_bench.runtime_images import load_images


def release():
    return {
        "schema": runtime.SCHEMA,
        "version": "1.0.0",
        "platform": "linux/amd64",
        "snapshot_image": "ghcr.io/example/simreadybench-runtime@sha256:" + "a" * 64,
        "images": {role: "sha256:" + value * 64 for role, value in zip(("actor", "verifier", "cpu"), "123")},
    }


def write_release(path, value=None):
    path.write_text(json.dumps(release() if value is None else value), encoding="utf-8")
    return path


@pytest.fixture
def service(monkeypatch):
    daytona = pytest.importorskip("daytona")
    monkeypatch.setenv("DAYTONA_API_KEY", "test-daytona-secret")

    class Snapshots:
        def __init__(self):
            self.values = {}
            self.created = []
            self.failure = None
            self.conflict = False

        def get(self, name):
            if self.failure:
                raise self.failure
            if name not in self.values:
                raise daytona.DaytonaNotFoundError("Missing snapshot")
            return self.values[name]

        def create(self, params, *, timeout):
            self.created.append((params, timeout))
            selected = SimpleNamespace(
                id="snapshot-id", name=params.name, image_name=params.image, state="active"
            )
            self.values[params.name] = selected
            if self.conflict:
                raise daytona.DaytonaConflictError("Created concurrently")
            return selected

    selected = Snapshots()
    monkeypatch.setattr(daytona, "Daytona", lambda config: SimpleNamespace(snapshot=selected))
    return selected


@pytest.mark.parametrize("damage", [
    "schema", "platform", "tag", "bare-image", "missing-role", "tagged-role", "extra", "version"
])
def test_release_rejects_unpinned_or_incompatible_runtime_metadata(damage):
    value = release()
    if damage in ("schema", "platform", "version"):
        value[damage] = {"schema": "other/v1", "platform": "linux/arm64", "version": "latest"}[damage]
    elif damage == "tag":
        value["snapshot_image"] = "ghcr.io/example/simreadybench-runtime:latest"
    elif damage == "bare-image":
        value["snapshot_image"] = "sha256:" + "a" * 64
    elif damage == "missing-role":
        value["images"].pop("verifier")
    elif damage == "tagged-role":
        value["images"]["cpu"] = "cpu:latest"
    else:
        value["unknown"] = True
    with pytest.raises(ValueError):
        runtime.validate_release(value)


def test_default_release_uses_packaged_bytes_and_missing_resource_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="--release-manifest"):
        runtime.load_release()
    path = tmp_path / runtime.RESOURCE
    path.parent.mkdir(parents=True)
    write_release(path)
    assert runtime.load_release() == release()


def test_setup_imports_pinned_outer_image_and_writes_loadable_mapping(tmp_path, service):
    source = write_release(tmp_path / "release.json")
    output = tmp_path / "configuration" / "runtime-images.json"
    first = runtime.setup_daytona(output=output, release_manifest=source)
    assert first["status"] == "created" and first["snapshot_id"] == "snapshot-id"
    assert first["snapshot"] == "simready-bench-cpu-1.0.0"
    assert load_images(output) == {**release()["images"], "snapshot": first["snapshot"]}
    params, timeout = service.created[0]
    assert params.image == release()["snapshot_image"]
    assert (params.resources.cpu, params.resources.memory, params.resources.disk) == (2, 6, 10)
    assert timeout == 600
    original = output.read_bytes()
    again = runtime.setup_daytona(output=output, release_manifest=source)
    assert again["status"] == "reused" and len(service.created) == 1
    assert output.read_bytes() == original
    assert not list(output.parent.glob(".runtime-images-*"))


def test_setup_retains_custom_snapshot_and_import_resources(tmp_path, service):
    source = write_release(tmp_path / "release.json")
    result = runtime.setup_daytona(
        output=tmp_path / "runtime.json", release_manifest=source, snapshot="team-runtime",
        cpu=4, memory=8, disk=20, timeout=900,
    )
    params, timeout = service.created[0]
    assert result["snapshot"] == "team-runtime" and params.name == "team-runtime"
    assert (params.resources.cpu, params.resources.memory, params.resources.disk) == (4, 8, 20)
    assert timeout == 900


def test_documented_parent_relative_paths_import_and_reuse_runtime(tmp_path, service, monkeypatch, capsys):
    from simready_bench.cli import main

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    write_release(tmp_path / "release.json")
    monkeypatch.chdir(checkout)
    command = [
        "setup", "daytona", "--output", "../simreadybench-runtime.json",
        "--release-manifest", "../release.json",
    ]
    assert main(command) == 0
    target = tmp_path / "simreadybench-runtime.json"
    original = target.read_bytes()
    assert load_images(target) == {
        **release()["images"], "snapshot": "simready-bench-cpu-1.0.0",
    }
    assert main(command) == 0 and len(service.created) == 1
    assert target.read_bytes() == original
    assert str(target) in capsys.readouterr().out


@pytest.mark.parametrize("selected", ["output", "release", "collapsed-parent"])
def test_parent_relative_normalisation_preserves_symlink_rejection(tmp_path, service, monkeypatch, selected):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = write_release(tmp_path / "release.json")
    destination = tmp_path / "destination"
    destination.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(destination, target_is_directory=True)
    monkeypatch.chdir(checkout)
    output = "../linked/runtime.json" if selected == "output" else "../runtime.json"
    manifest = "../release.json"
    if selected == "release":
        (tmp_path / "release-link.json").symlink_to(source)
        manifest = "../release-link.json"
    elif selected == "collapsed-parent":
        output = "../linked/../runtime.json"
    with pytest.raises(ValueError, match="symbolic links"):
        runtime.setup_daytona(output=output, release_manifest=manifest)
    assert not (tmp_path / "runtime.json").exists()
    assert not (destination / "runtime.json").exists() and not service.created


@pytest.mark.parametrize("damage", ["image", "state", "name"])
def test_setup_refuses_an_existing_incompatible_snapshot(tmp_path, service, damage):
    name = "simready-bench-cpu-1.0.0"
    selected = SimpleNamespace(name=name, image_name=release()["snapshot_image"], state="active")
    if damage == "image":
        selected.image_name = "ghcr.io/example/other@sha256:" + "b" * 64
    elif damage == "state":
        selected.state = "building"
    else:
        selected.name = "other"
    service.values[name] = selected
    output = tmp_path / "runtime.json"
    with pytest.raises((ValueError, RuntimeError)):
        runtime.setup_daytona(output=output, release_manifest=write_release(tmp_path / "release.json"))
    assert not output.exists() and not service.created


def test_active_enum_and_concurrent_matching_snapshot_are_reused(tmp_path, service):
    class State(Enum):
        ACTIVE = "active"

    service.conflict = True
    source = write_release(tmp_path / "release.json")
    output = tmp_path / "runtime.json"
    result = runtime.setup_daytona(output=output, release_manifest=source)
    assert result["status"] == "reused"
    service.values[result["snapshot"]].state = State.ACTIVE
    assert runtime.setup_daytona(output=output, release_manifest=source)["status"] == "reused"
    assert len(service.created) == 1


def test_setup_preserves_conflicting_output_before_contacting_daytona(tmp_path, service):
    output = tmp_path / "runtime.json"
    output.write_text('{"existing": true}')
    with pytest.raises(FileExistsError, match="Preserving"):
        runtime.setup_daytona(output=output, release_manifest=write_release(tmp_path / "release.json"))
    assert output.read_text() == '{"existing": true}' and not service.created


def test_setup_requires_credential_and_does_not_treat_auth_failure_as_missing(tmp_path, service, monkeypatch):
    source = write_release(tmp_path / "release.json")
    monkeypatch.delenv("DAYTONA_API_KEY")
    with pytest.raises(ValueError, match="Set DAYTONA_API_KEY"):
        runtime.setup_daytona(output=tmp_path / "runtime.json", release_manifest=source)
    monkeypatch.setenv("DAYTONA_API_KEY", "test-daytona-secret")
    from daytona import DaytonaAuthenticationError

    service.failure = DaytonaAuthenticationError("Rejected test-daytona-secret")
    with pytest.raises(RuntimeError) as captured:
        runtime.setup_daytona(output=tmp_path / "runtime.json", release_manifest=source)
    assert "test-daytona-secret" not in str(captured.value)
    assert "[redacted]" in str(captured.value) and not service.created


def test_output_changed_during_import_is_preserved(tmp_path, service, monkeypatch):
    source = write_release(tmp_path / "release.json")
    output = tmp_path / "runtime.json"
    original = service.create

    def create(params, *, timeout):
        selected = original(params, timeout=timeout)
        output.write_text('{"concurrent": true}')
        return selected

    monkeypatch.setattr(service, "create", create)
    with pytest.raises(FileExistsError, match="Preserving"):
        runtime.setup_daytona(output=output, release_manifest=source)
    assert output.read_text() == '{"concurrent": true}'


def test_setup_cli_dispatches_without_requiring_a_model_or_benchmark(monkeypatch, capsys, tmp_path):
    from simready_bench.cli import main

    calls = []

    def setup(**options):
        calls.append(deepcopy(options))
        return {"snapshot": "team-runtime", "runtime_images": str(options["output"])}

    monkeypatch.setattr(runtime, "setup_daytona", setup)
    assert main([
        "setup", "daytona", "--output", str(tmp_path / "runtime.json"),
        "--snapshot", "team-runtime", "--release-manifest", "release.json",
    ]) == 0
    assert calls[0]["release_manifest"] == Path("release.json")
    assert calls[0]["snapshot"] == "team-runtime"
    assert "Snapshot ready: team-runtime" in capsys.readouterr().out


def test_setup_cli_returns_failure_without_claiming_ready(monkeypatch, capsys):
    from simready_bench.cli import main

    def fail(**options):
        raise RuntimeError("Snapshot is not active")

    monkeypatch.setattr(runtime, "setup_daytona", fail)
    assert main(["setup", "daytona", "--output", "runtime.json"]) == 2
    captured = capsys.readouterr()
    assert not captured.out and captured.err == "Snapshot is not active\n"
