"""Local Hub handoff finalizes package hashes without publishing or asset fetches."""
import copy
import hashlib
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

from test_export import REPO, REVISION, release_input, wheel, synthetic_public_download


def tool():
    source = Path("native/tools/prepare_hub_release.py")
    assert source.is_file(), "clean Hub handoff helper is missing"
    spec = importlib.util.spec_from_file_location("prepare_hub_release", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_input():
    release = release_input(100)
    for index, task in enumerate(release["tasks"]):
        original = task["id"]
        task["object_name"] = f"workshop clamp {index}"
        task["id"] = f"nv100_stc_source_{index:03}"
        for role in ("actor", "verifier"):
            task[role]["task_id"] = task["id"]
            for file in task[role]["files"]:
                file["path"] = file["path"].replace("/" + original + "/", "/" + task["id"] + "/")
    return release


def test_owner_acknowledgment_is_required_before_metadata_or_export(tmp_path, wheel):
    with pytest.raises(ValueError, match="asset.freeze"):
        tool().prepare_hub_release(canonical_input(), wheel, tmp_path / "release",
                                   metadata_getter=lambda _: pytest.fail("unacknowledged release requested metadata"))
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("metadata", [
    {"id": REPO, "sha": "2" * 40, "private": False, "gated": False},
    {"id": "another/repo", "sha": REVISION, "private": False, "gated": False},
    {"id": REPO, "sha": REVISION, "private": True, "gated": False},
    {"id": REPO, "sha": REVISION, "private": False},
    *[{"id": REPO, "sha": REVISION, "private": False, "gated": gated}
      for gated in (None, "auto", "manual")],
])
def test_wrong_or_nonpublic_hf_revision_cannot_create_handoff(tmp_path, wheel, metadata):
    with pytest.raises(ValueError):
        tool().prepare_hub_release(canonical_input(), wheel, tmp_path / "release", asset_freeze_verified=True,
                                   metadata_getter=lambda _: metadata)
    assert not (tmp_path / "release").exists()


def test_verified_freeze_finalizes_100_task_handoff_with_consistent_root_hashes(tmp_path, wheel):
    from harbor.models.dataset.manifest import DatasetManifest
    from harbor.models.task.task import Task
    from harbor.publisher.packager import Packager

    release = canonical_input()
    original = copy.deepcopy(release)
    requested = []

    def metadata(url):
        requested.append(url)
        return {"id": REPO, "sha": REVISION, "private": False, "gated": False}

    output = tmp_path / "release"
    receipt = tool().prepare_hub_release(release, wheel, output, asset_freeze_verified=True, metadata_getter=metadata)
    assert release == original
    assert requested == [f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}"]
    dataset = DatasetManifest.from_toml_file(output / "dataset.toml")
    assert dataset.schema_version == "1.0" and len(dataset.tasks) == 100
    assert tomllib.loads((output / "dataset.toml").read_text())["dataset"]["name"] == REPO
    expected_names = {"hcltech-robotics/" + task["id"] for task in release["tasks"]}
    assert {task.name for task in dataset.tasks} == expected_names
    for ref in dataset.tasks:
        task_path = output / ref.name.split("/", 1)[1]
        task = Task(task_path)
        assert task.config.schema_version == "1.4"
        supplied = next(row for row in release["tasks"] if "hcltech-robotics/" + row["id"] == ref.name)
        assert (task_path / "instruction.md").read_text().startswith("Reconstruct this " + supplied["object_name"] + " ")
        assert (task_path / "environment/instruction.md").read_bytes() == b"x"
        assert task.name == ref.name
        assert ref.digest == "sha256:" + Packager.compute_content_hash(task_path)[0]
    for file in dataset.files:
        assert file.digest == "sha256:" + hashlib.sha256((output / file.path).read_bytes()).hexdigest()
    status = json.loads((output / "release.json").read_text())["export"]
    assert status == {"fixture": False}
    readme = (output / "README.md").read_text()
    assert readme.startswith("# SimReadyBench NV100\n")
    for expected in ("100 tasks", "100 public observation images", "400 comparison images",
                     "harbor run --dataset", "1 GiB", "10000 files", "five", "95 * passed / 46",
                     "51 optional", "sqrt(G * R)", "Kinematic", "MIT", REVISION):
        assert expected in readme
    for unwanted in ("stc/nv100", "frozen nv100 release", "ready for publication", "release owner", "acknowledg", "handoff", "audit", "unverified", "oracle", "not publish ready"):
        assert unwanted not in readme.lower()
    assert "## Frozen task observations" in readme
    assert readme.count("[preview/download]") == 100
    assert "harbor run --dataset hcltech-robotics/simreadybench " in readme
    assert len(list(output.glob("*/environment/observation.png"))) == 100
    assert len(list(output.glob("*/environment/instruction.md"))) == 100
    assert receipt["ready_for_publication"] is True
    assert receipt["published_consumption_verified"] is False
    assert receipt["task_count"] == 100
    assert receipt["dataset_digest"] == "sha256:" + dataset.compute_content_hash()
    inventory = json.loads((output / "inventory.json").read_text())
    actual_files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert actual_files == {item["path"] for item in inventory} | {"inventory.json"}
    for item in inventory:
        data = (output / item["path"]).read_bytes()
        assert item["size_bytes"] == len(data)
        assert item["sha256"] == hashlib.sha256(data).hexdigest()
    assert {path.name for path in output.iterdir() if path.is_dir()} == {task["id"] for task in release["tasks"]}
    assert not any(path.is_symlink() for path in output.rglob("*"))


def test_helper_rejects_even_consistently_rehashed_wrong_dataset_name(tmp_path, wheel):
    module = tool()
    output = tmp_path / "release"
    release = canonical_input()
    module.prepare_hub_release(release, wheel, output, asset_freeze_verified=True,
        metadata_getter=lambda _: {"id": REPO, "sha": REVISION, "private": False, "gated": False})
    manifest = output / "dataset.toml"
    text = manifest.read_text()
    name = tomllib.loads(text)["dataset"]["name"]
    manifest.write_text(text.replace('name = "' + name + '"', 'name = "hcltech-robotics/wrong-dataset"', 1))
    # Keep the inventory valid, so only the explicit destination guard can reject.
    inventory = json.loads((output / "inventory.json").read_text())
    for row in inventory:
        if row["path"] == "dataset.toml":
            data = manifest.read_bytes();row.update(size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    module._write_json(output / "inventory.json", inventory)
    with pytest.raises(ValueError, match="dataset package name"):
        module._validate_tree(output, release, wheel.name)
