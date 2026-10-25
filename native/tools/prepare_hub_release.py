#!/usr/bin/env python3
"""Prepare a clean local STC/NV100 Hub handoff; never publish or upload.

Run in an environment containing the accepted evaluator wheel and Harbor 0.24.0.
The owner flag acknowledges the completed 100-task/500-frame freeze, licenses,
and checksums. It is owner evidence, not an asset audit performed by this tool.
Remote reads are the public HF revision metadata and the two declared actor
inputs per task. A verified local mirror can replace the actor input downloads.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from simready_eval.export import _fresh_output, _validate_release, export_release
from simready_eval.fetch import REPO_ID

ROOT_FILES = {"dataset.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "release.json", "metric.py", "inventory.json"}
DATASET_FILES = ROOT_FILES - {"dataset.toml", "inventory.json"}
TASK_FILES = {"instruction.md", "task.toml", "environment/Dockerfile", "environment/Dockerfile.feedback",
              "environment/docker-compose.yaml", "environment/actor-assets.json", "environment/actor.requirements.txt",
              "environment/observation.png", "environment/instruction.md",
              "environment/feedback.requirements.txt", "tests/Dockerfile", "tests/verifier-assets.json",
              "tests/verifier.requirements.txt", "tests/evaluator.json", "tests/test.sh"}
STATUS = {"fixture": False, "release_owner_asset_freeze_verified": True, "ready_for_publication": True,
          "published_consumption_verified": False, "task_specific_oracle_not_supplied": True}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _public_metadata(url):
    request = Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache",
                                    "User-Agent": "simready-harbor-handoff/1"})
    with urlopen(request, timeout=30) as response:
        payload = response.read(8 * 1024 * 1024 + 1)
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("public HF metadata response exceeds the size limit")
    return json.loads(payload)


def _readme(release):
    return f'''# Frozen NV100 release

This frozen 100-task STC/NV100 package is ready for publication following the
release owner's acknowledgment of the completed asset freeze: 500 observation
and comparison frames, camera data, licenses, and checksums. Its canonical public
HF dataset is `{release['repo_id']}` at full revision `{release['revision']}`.
That revision's public identity was checked while preparing this local handoff.

Run the tasks with standard Harbor 0.24.0. Setup fetches only each role's pinned
HF verifier assets; final scoring is offline. Each task package includes its frozen
public observation PNG and instruction bytes, verified against its role manifest. Runner provider endpoints belong in
`agents[].extra_allowed_hosts`. The agent phase begins with an empty allowlist,
so HF/Xet reference access stays blocked. Blender is reserved for the verifier
renderer and is prohibited for agent authoring.

Published consumption and runtime validation of all 100 tasks remain unverified.
No task-specific oracle has been supplied. The owner's freeze acknowledgment and
local package/schema/hash checks establish readiness for publication; this
helper performs no upload, publication, full asset re-audit, or renderer execution.
'''


def _inventory(root):
    return [{"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
             "sha256": _sha(path.read_bytes())}
            for path in sorted(root.rglob("*")) if path.is_file() and path != root / "inventory.json"]


def _validate_tree(root, release, wheel_name):
    from harbor.models.dataset.manifest import DatasetManifest
    from harbor.models.task.task import Task
    from harbor.publisher.packager import Packager

    names = {task["id"] for task in release["tasks"]}
    task_files = TASK_FILES | {"environment/" + wheel_name, "tests/" + wheel_name}
    expected = ROOT_FILES | {name + "/" + file for name in names for file in task_files}
    expected_dirs = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."}
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("handoff must not contain symlinks")
        if path.is_dir():
            if relative not in expected_dirs:
                raise ValueError("handoff contains an unexpected directory")
        elif path.is_file():
            actual.add(relative)
        else:
            raise ValueError("handoff contains a nonregular file")
    if actual != expected or {path.name for path in root.iterdir() if path.is_dir()} != names:
        raise ValueError("handoff must contain exactly the 100 source-ID task directories and allowed generated files")
    dataset = DatasetManifest.from_toml_file(root / "dataset.toml")
    expected_names = {"hcltech-robotics/" + name for name in names}
    if dataset.schema_version != "1.0" or len(dataset.tasks) != 100 or {task.name for task in dataset.tasks} != expected_names:
        raise ValueError("dataset schema or exact source-ID task names do not match the frozen release")
    for ref in dataset.tasks:
        task_root = root / ref.name.split("/", 1)[1]
        task = Task(task_root)
        if task.config.schema_version != "1.4" or task.name != ref.name:
            raise ValueError("task schema or package name does not match the dataset")
        if ref.digest != "sha256:" + Packager.compute_content_hash(task_root)[0]:
            raise ValueError("task publisher digest mismatch")
    if len(dataset.files) != len(DATASET_FILES) or {file.path for file in dataset.files} != DATASET_FILES:
        raise ValueError("dataset-level file references do not match the allowlist")
    for file in dataset.files:
        if file.digest != "sha256:" + _sha((root / file.path).read_bytes()):
            raise ValueError("dataset-level file digest mismatch")
    if json.loads((root / "inventory.json").read_text()) != _inventory(root):
        raise ValueError("handoff inventory mismatch")
    return dataset


def prepare_hub_release(release, wheel, output, *, asset_freeze_verified=False, metadata_getter=None, public_assets=None):
    """Finalize a local handoff after trusted owner acknowledgment.

    ``metadata_getter`` is an internal test transport seam accepting the exact
    public metadata URL. The CLI always performs the actual unauthenticated GET.
    Input remains the exporter's strict canonical release schema: draft export
    output/status fields are not accepted or silently stripped.
    """
    if asset_freeze_verified is not True:
        raise ValueError("--asset-freeze-verified owner acknowledgment is required")
    release = _validate_release(release, fixture=False)
    output = _fresh_output(output)
    if version("harbor") != "0.24.0":
        raise ValueError("handoff validation requires official Harbor 0.24.0")
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/revision/{release['revision']}"
    metadata = (metadata_getter or _public_metadata)(url)
    if (not isinstance(metadata, dict) or metadata.get("id") != REPO_ID or metadata.get("sha") != release["revision"]
            or metadata.get("private") is not False or metadata.get("gated") is not False):
        raise ValueError("declared revision must exist in the exact public ungated HF dataset and return its full SHA")
    wheel = Path(wheel)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".simready-hub-handoff-", dir=output.parent) as temporary:
        staged = Path(temporary) / "release"
        export_release(release, staged, wheel, fixture=False, public_assets=public_assets)
        finalized = {**release, "export": dict(STATUS)}
        _write_json(staged / "release.json", finalized)
        previews = (staged / "README.md").read_text().split("## Frozen task observations", 1)[1]
        (staged / "README.md").write_text(_readme(release) + "\n## Frozen task observations" + previews,
                                          encoding="utf-8")
        from harbor.models.dataset.manifest import DatasetManifest
        dataset = DatasetManifest.from_toml_file(staged / "dataset.toml")
        for file in dataset.files:
            file.digest = "sha256:" + _sha((staged / file.path).read_bytes())
        (staged / "dataset.toml").write_text('schema_version = "1.0"\n\n' + dataset.to_toml(), encoding="utf-8")
        _write_json(staged / "inventory.json", _inventory(staged))
        dataset = _validate_tree(staged, release, wheel.name)
        receipt = {"schema": "simready-hub-handoff-v1", "canonical": {"repo_id": REPO_ID, "revision": release["revision"]},
                   **STATUS, "task_count": len(dataset.tasks), "harbor_version": "0.24.0",
                   "dataset_digest": "sha256:" + dataset.compute_content_hash(),
                   "inventory_sha256": _sha((staged / "inventory.json").read_bytes()),
                   "wheel_sha256": _sha(wheel.read_bytes()), "hf_revision_metadata_verified": True}
        _fresh_output(output)
        staged.rename(output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-freeze-verified", action="store_true")
    parser.add_argument("--public-assets", type=Path, help="Optional local mirror of the pinned public HF paths")
    args = parser.parse_args(argv)
    try:
        receipt = prepare_hub_release(json.loads(args.release.read_text()), args.wheel, args.output,
                                      asset_freeze_verified=args.asset_freeze_verified, public_assets=args.public_assets)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Hub handoff preparation failed: {error}\n")
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
