"""A mutable bucket mirror preserves pinned canonical bytes without I/O."""
import copy
import hashlib
import importlib
import json
import subprocess
import sys

import pytest

from test_export import REPO, REVISION, release_input


def mirror():
    assert importlib.util.find_spec("simready_eval.mirror") is not None, "set mirror manifest helper is missing"
    return importlib.import_module("simready_eval.mirror")


def test_mirror_preserves_exact_declared_bytes_and_set_relative_keys():
    release = release_input(2)
    result = mirror().build_mirror_manifest(release, fixture=True)
    assert result["schema"] == "simready-set-mirror-v1"
    assert result["set_id"] == "nv100"
    assert result["canonical"] == {"repo_id": REPO, "revision": REVISION}
    assert "mutable_not_authority" in result["warnings"]
    expected = {file["path"]: file for task in release["tasks"] for role in ("actor", "verifier") for file in task[role]["files"]}
    assert len(result["files"]) == len(expected)
    for entry in result["files"]:
        source = expected[entry["source_path"]]
        assert entry["destination_key"] == entry["source_path"].removeprefix("benchmark/nv100/v1/")
        assert entry["sha256"] == source["sha256"]
        assert entry["size_bytes"] == source["size_bytes"]
    unhashed = {key: value for key, value in result.items() if key != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert result["manifest_sha256"] == digest
    assert result == mirror().build_mirror_manifest(copy.deepcopy(release), fixture=True)


def test_explicit_set_metadata_deduplicates_identical_bytes_and_rejects_conflicts():
    file = {"path": "benchmark/nv100/v1/LICENSE", "size_bytes": 4, "sha256": "a" * 64}
    result = mirror().build_mirror_manifest(release_input(), [file, dict(file)], fixture=True)
    assert sum(entry["destination_key"] == "LICENSE" for entry in result["files"]) == 1
    conflicting = {**file, "sha256": "b" * 64}
    with pytest.raises(ValueError):
        mirror().build_mirror_manifest(release_input(), [file, conflicting], fixture=True)


@pytest.mark.parametrize("path", ["results/README.md", "viewer/README.md", "benchmark/other/v1/README.md", "benchmark/nv100/v1/../README.md", "benchmark/nv100/v1/private.md", "benchmark/nv100/v1/orichalcum/README.md", "benchmark/nv100/v1/tasks/other/public/observation.png", "benchmark/nv100/v1/unknown.json"])
def test_mirror_rejects_unsafe_unrelated_or_unknown_extras(path):
    with pytest.raises(ValueError):
        mirror().build_mirror_manifest(release_input(), [{"path": path, "size_bytes": 1, "sha256": "a" * 64}], fixture=True)


def test_mirror_rejects_cross_revision_or_legacy_task_files():
    release = release_input()
    release["tasks"][0]["actor"]["revision"] = "3" * 40
    with pytest.raises(ValueError):
        mirror().build_mirror_manifest(release, fixture=True)
    release = release_input()
    release["tasks"][0]["verifier"]["files"][0]["path"] = "results/runs/private.usd"
    with pytest.raises(ValueError):
        mirror().build_mirror_manifest(release, fixture=True)


def test_cli_writes_local_manifest_without_provisioning(tmp_path):
    release = tmp_path / "release.json"
    release.write_text(json.dumps(release_input()))
    extra = tmp_path / "extras.json"
    extra.write_text(json.dumps([{"path": "benchmark/nv100/v1/checksums.json", "size_bytes": 2, "sha256": "a" * 64}]))
    output = tmp_path / "mirror.json"
    result = subprocess.run([sys.executable, "-m", "simready_eval.mirror", "--release", str(release), "--output", str(output), "--extra-files", str(extra), "--fixture"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text())
    assert "bucket_name_chosen_later" in manifest["warnings"]
    assert "server_side_xet_copy_not_performed" in manifest["warnings"]
    assert set(tmp_path.iterdir()) == {release, extra, output}
