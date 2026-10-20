from pathlib import Path
import importlib.util
import io
import json
import gzip
import hashlib
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actor_and_verifier_stage_current_clients_without_source_duplicates(tmp_path):
    build = load("container_build", "containers/build.py")
    context = build.prepare_context("actor", tmp_path / "actor")
    for name in (
        "actor_mcp.py",
        "actor_relay.py",
        "pi_runtime.py",
        "verifier.py",
        "nvidia_profile.json",
    ):
        source = ROOT / "src/simready_bench/engine" / name
        assert (context / name).read_bytes() == source.read_bytes()
        assert not (ROOT / "containers/actor" / name).exists()
    assert "--target" in build.build_command(
        "actor", tag="simreadybench/actor:1.0.0", context=context
    )
    assert "verifier" in build.build_command(
        "verifier", tag="simreadybench/verifier:1.0.0", context=context
    )


def test_cpu_context_uses_supplied_wheel_and_runtime_lock(tmp_path):
    build = load("container_build", "containers/build.py")
    wheel = tmp_path / "simready_bench-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel bytes")
    context = build.prepare_context("cpu", tmp_path / "cpu", wheel=wheel)
    assert (context / "wheels" / wheel.name).read_bytes() == wheel.read_bytes()
    assert (context / "runtime.txt").read_bytes() == (
        ROOT / "locks/runtime.txt"
    ).read_bytes()
    assert "wheels/*.whl" in (context / "Dockerfile").read_text()
    assert (
        'org.opencontainers.image.version="1.0.0"'
        in (context / "Dockerfile").read_text()
    )
    with pytest.raises(ValueError, match="--wheel"):
        build.prepare_context("cpu", tmp_path / "missing")


def test_verifier_client_runs_with_only_staged_files(tmp_path):
    build = load("container_build", "containers/build.py")
    context = build.prepare_context("verifier", tmp_path / "verifier")
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(context / "verifier.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--socket" in result.stdout and "--output-dir" in result.stdout
    profile = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import runpy, sys\nclient = runpy.run_path(sys.argv[1])\nprint(len(client['pinned_profile']()['mandatory_requirements']))",
            str(context / "verifier.py"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert profile.returncode == 0, profile.stderr
    assert profile.stdout.strip() == "46"


def saved_images(change=None):
    blobs = {}

    def blob(value):
        raw = json.dumps(value, sort_keys=True).encode()
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        blobs["blobs/sha256/" + digest.removeprefix("sha256:")] = raw
        return digest, len(raw)

    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w"):
        pass
    layer_raw = layer.getvalue()
    layer_digest = "sha256:" + hashlib.sha256(layer_raw).hexdigest()
    blobs["blobs/sha256/" + layer_digest.removeprefix("sha256:")] = layer_raw
    producer, consumer, tags, descriptors = {}, {}, {}, []
    for role in ("actor", "verifier", "cpu"):
        config, config_size = blob({
            "architecture": "amd64", "os": "linux", "config": {"Labels": {"role": role}},
            "rootfs": {"type": "layers", "diff_ids": [layer_digest]},
        })
        producer[role] = config
        tag = "simreadybench/snapshot-" + role + ":" + config.removeprefix("sha256:")
        tags[role] = tag
        manifest, size = blob({
            "schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": config, "size": config_size},
            "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar", "digest": layer_digest, "size": len(layer_raw)}],
        })
        consumer[role] = manifest
        descriptors.append({
            "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": manifest, "size": size,
            "annotations": {"io.containerd.image.name": "docker.io/" + tag},
        })
    if change == "untagged":
        descriptors = [{key: value for key, value in descriptors[0].items() if key != "annotations"}]
    elif change == "missing":
        descriptors.pop()
    elif change == "duplicate":
        descriptors.append(dict(descriptors[0]))
    elif change == "unexpected":
        descriptors[0]["annotations"]["io.containerd.image.name"] = "docker.io/other/image:latest"
    elif change == "digest":
        descriptors[0]["digest"] = "sha256:" + "0" * 64
    elif change == "blob":
        path = "blobs/sha256/" + consumer["actor"].removeprefix("sha256:")
        blobs[path] += b"\n"
    blobs["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
    blobs["index.json"] = json.dumps({"schemaVersion": 2, "manifests": descriptors}).encode()
    saved = io.BytesIO()
    with tarfile.open(fileobj=saved, mode="w") as archive:
        for name, raw in blobs.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return producer, consumer, tags, saved.getvalue(), layer_digest


def mock_docker_export(monkeypatch, prepare, producer, raw):
    calls, tagged = [], []
    monkeypatch.setattr(prepare.subprocess, "check_output", lambda command, text: producer[command[3]] + "\n")
    monkeypatch.setattr(prepare.subprocess, "run", lambda command, check: tagged.append(command))

    class Process:
        def __init__(self):
            self.stdout = io.BytesIO(raw)
            self.stderr = io.BytesIO(b"")

        def wait(self):
            return 0

        def poll(self):
            return 0

    def save(command, **kwargs):
        calls.append(command)
        return Process()

    monkeypatch.setattr(prepare.subprocess, "Popen", save)
    return calls, tagged


def test_snapshot_exports_named_images_and_uses_consumer_manifest_identities(tmp_path, monkeypatch):
    prepare = load("snapshot_prepare", "containers/daytona/prepare_snapshot.py")
    producer, consumer, tags, saved, shared_layer = saved_images()
    calls, tagged = mock_docker_export(monkeypatch, prepare, producer, saved)
    output = tmp_path / "snapshot"
    actual = prepare.prepare(
        actor_image="actor", verifier_image="verifier", cpu_image="cpu", output=output
    )
    assert actual == consumer and producer != consumer
    assert json.loads((output / "runtime-images.json").read_text()) == consumer
    assert json.loads((output / "local-runtime-images.json").read_text()) == producer
    assert (output / "images.txt").read_text().splitlines() == list(consumer.values())
    assert tagged == [["docker", "image", "tag", producer[role], tag] for role, tag in tags.items()]
    assert calls == [["docker", "save", *tags.values()]]
    assert gzip.decompress((output / "images.tar.gz").read_bytes()) == saved
    with tarfile.open(output / "images.tar.gz", "r:gz") as archive:
        assert archive.getnames().count("blobs/sha256/" + shared_layer.removeprefix("sha256:")) == 1
    assert {path.name for path in output.iterdir()} == {
        "images.tar.gz",
        "images.txt",
        "runtime-images.json",
        "local-runtime-images.json",
        "snapshot.Dockerfile",
        "bootstrap.sh",
    }
    assert (
        "COPY runtime-images.json /opt/simready/runtime-images.json"
        in (output / "snapshot.Dockerfile").read_text()
    )


@pytest.mark.parametrize("change", ["untagged", "missing", "duplicate", "unexpected", "digest", "blob"])
def test_snapshot_refuses_incomplete_or_inconsistent_oci_export(tmp_path, monkeypatch, change):
    prepare = load("snapshot_prepare", "containers/daytona/prepare_snapshot.py")
    producer, _, _, saved, _ = saved_images(change)
    mock_docker_export(monkeypatch, prepare, producer, saved)
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError):
        prepare.prepare(actor_image="actor", verifier_image="verifier", cpu_image="cpu", output=output)
    assert not (output / "runtime-images.json").exists()
    assert not (output / "local-runtime-images.json").exists()
    assert not (output / "images.txt").exists()


def test_runtime_and_build_locks_keep_hashed_dependencies():
    for name in ("runtime.txt", "build.txt"):
        text = (ROOT / "locks" / name).read_text()
        assert "--hash=sha256:" in text
        assert "#" not in text
        assert not any(
            value in text for value in ("/tmp/", "/home/", "file://", "git+")
        )
