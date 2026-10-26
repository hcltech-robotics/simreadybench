"""Prepare one explicitly qualified asset closure using only the neutral wheel.

The source must be an entrypoint within package_root. Every non-license
file must be reachable through authored dependency paths; collection roots and
external dependencies are rejected. The entrypoint's original name and bytes
are preserved. No renderer output is qualified until the transaction completes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

import numpy as np
from PIL import Image

from . import correspondence, render
from .package import EXEC_EXTS, REFERENCE_LIMIT_BYTES, LIMIT_FILES, USD_EXTS, inspect_package
from .limits import CANDIDATE_LABEL
from .usd import observe

SCHEMA = "simready-asset-preparation-v1"
INVENTORY_SCHEMA = "simready-asset-inventory-v1"
FORBIDDEN_NAMES = {".git", ".venv", ".aws", ".codex", "__pycache__", "node_modules",
                   "scores.json", "reward.txt", "rewards.json", "results.json"}


class _Reject(ValueError):
    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _row(path, root):
    return {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": _hash(path)}


def _forbidden(path):
    return any(part.casefold() in FORBIDDEN_NAMES or
               "orichalcum" in part.casefold() or "oldresults" in part.casefold()
               for part in Path(path).parts)


def _scan(root):
    """No-follow bounded scan; never recurse outside this declared package."""
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise _Reject("qualified_package_root_required")
    files, total, entries = [], 0, 0
    def walk(directory):
        nonlocal total, entries
        with os.scandir(directory) as listing:
            for item in sorted(listing, key=lambda item: item.name):
                entries += 1
                if entries > LIMIT_FILES:
                    raise _Reject("package_entry_limit_exceeded")
                path = Path(item.path)
                relative = path.relative_to(root)
                if _forbidden(relative):
                    raise _Reject("legacy_or_result_content_forbidden", path=str(relative))
                mode = item.stat(follow_symlinks=False)
                if stat.S_ISDIR(mode.st_mode):
                    walk(path)
                elif not stat.S_ISREG(mode.st_mode) or mode.st_nlink != 1:
                    raise _Reject("nonregular_or_linked_file", path=str(relative))
                else:
                    if mode.st_mode & 0o111 or path.suffix.casefold() in EXEC_EXTS:
                        raise _Reject("executable_file_forbidden", path=str(relative))
                    total += mode.st_size
                    if total > REFERENCE_LIMIT_BYTES:
                        raise _Reject("package_byte_limit_exceeded")
                    # These exact legacy markers are forbidden even if renamed;
                    # scanning bytes neither imports nor executes package content.
                    with path.open("rb") as stream:
                        tail = b""
                        while chunk := stream.read(1024 * 1024):
                            value = (tail + chunk).lower()
                            if b"orichalcum" in value or b"oldresults" in value:
                                raise _Reject("legacy_or_result_content_forbidden", path=str(relative))
                            tail = value[-16:]
                    files.append(path)
    walk(root)
    return files


def _closure(source, root, files, *, strict=True):
    """Find authored dependencies without composing or modifying source layers."""
    from pxr import Sdf, UsdUtils
    import xml.etree.ElementTree as ET
    available = set(files)
    def dependency(raw, owner):
        raw = raw.split("[", 1)[0]
        if "<UDIM>" in raw:
            path = owner.parent / raw
            matches = [tile.resolve() for tile in path.parent.glob(
                path.name.replace("<UDIM>", "[0-9][0-9][0-9][0-9]"))]
            if not matches:
                raise _Reject("unrecorded_dependency", path=str(path))
            return matches
        return [(owner.parent / raw).resolve()] if raw else []
    def dependencies_of(path):
        dependencies = []
        suffix = path.suffix.casefold()
        if suffix in USD_EXTS:
            layer = Sdf.Layer.OpenAsAnonymous(str(path))
            def collect(value):
                dependencies.extend(dependency(value, path))
                return value
            UsdUtils.ModifyAssetPaths(layer, collect)
        elif suffix == ".mtlx":
            for element in ET.fromstring(path.read_bytes()).iter():
                for name, value in element.attrib.items():
                    if name in ("file", "href", "filename") or (name == "value" and element.attrib.get("type") == "filename"):
                        dependencies.extend(dependency(value, path))
        elif suffix == ".mdl":
            for value in re.findall(r'(?:texture_\w+|light_profile|bsdf_measurement)\s*\(\s*"([^"\n]+)"', path.read_text()):
                dependencies.extend(dependency(value, path))
        return dependencies
    reachable, pending = set(), [source]
    while pending:
        path = pending.pop()
        if path in reachable:
            continue
        if path not in available:
            raise _Reject("unrecorded_dependency", path=str(path))
        reachable.add(path)
        pending.extend(dependencies_of(path))
    def notice(path):
        return path.name.casefold().startswith(("license", "notice", "copying", "copyright", "readme", "provenance"))
    unrelated = [p.relative_to(root).as_posix() for p in files if p not in reachable and not notice(p)]
    if strict and unrelated:
        raise _Reject("unqualified_package_closure", unreferenced_files=unrelated)
    return reachable | {path for path in files if notice(path)}


def runtime_closure(source, package_root):
    """Inventory one original package without copying its unrelated archive.

    Preserve authored references (including referenced archives and UDIM tiles)
    and source notices. The returned paths map directly to original files; after
    selective materialization, run the normal native package validation before
    eligibility generation. This function does not render or publish anything.
    """
    source, root = Path(source).absolute(), Path(package_root).absolute()
    if not source.is_relative_to(root) or source.resolve() != source:
        raise _Reject("qualified_entrypoint_required")
    files = _scan(root)
    selected = _closure(source, root, files, strict=False)
    return {"schema": "simready-runtime-closure-v1",
            "source_entrypoint": source.relative_to(root).as_posix(),
            "files": [_row(path, root) for path in sorted(selected)],
            "excluded_files": sorted(path.relative_to(root).as_posix()
                                     for path in files if path not in selected),
            "qualification": "dependency_inventory_only; validate selective materialization before evaluation"}


def public_instruction(dimensions):
    """Retain the approved public dimensions without needing another render."""
    if (not isinstance(dimensions, list) or len(dimensions) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not 1e-9 <= value <= 1e6 for value in dimensions)):
        raise ValueError("public_nominal_dimensions_required")
    return (
        "Reconstruct the asset shown in observation.png as a self-contained USD package.\n\n"
        "Nominal dimensions in canonical metres XYZ: " + json.dumps(dimensions) + "\n\n"
        "Preserve the visible geometry, materials, rigid bodies, joints and articulation. "
        "Write a top-level asset.usd and include all relative dependencies in your submission.\n\n"
        f"Submit only USD and its data dependencies: at most {CANDIDATE_LABEL} and {LIMIT_FILES} files/directories combined. "
        "Keep authoring programs outside the submission; links, executable files and external dependencies are not accepted.\n\n"
        "Do not use Blender for authoring; Blender is used only by the verifier renderer.\n")


def freeze_cameras(observation, *, resolution=512, task_id=None):
    """Freeze five reference-only orthographic views in canonical metres/Z-up.

    Vertical scale is 1.65 times the largest bounding-box span. Camera distance
    is three spans. The separate observation faces (1,-1,.8); comparison views
    face isometric (1,1,.8), front (-Y), side (+X), and top (+Z), sharing a center.
    Matrix columns are right/up/backward/position, acting on column vectors.
    Supplying task_id replaces only the observation with the seeded policy.
    """
    triangles = np.asarray(observation["triangles"], dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not triangles.size or not np.isfinite(triangles).all():
        raise _Reject("finite_reference_surface_required")
    points = triangles.reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    center, span = (low + high) / 2, float(np.max(high - low))
    if not math.isfinite(span) or span <= 0:
        raise _Reject("nondegenerate_reference_bounds_required")
    views = []
    for name, direction, up in (("isometric", [1., 1., .8], [0., 0., 1.]),
                                ("front", [0., -1., 0.], [0., 0., 1.]),
                                ("side", [1., 0., 0.], [0., 0., 1.]),
                                ("top", [0., 0., 1.], [0., 1., 0.]),
                                ("observation", [1., -1., .8], [0., 0., 1.])):
        backward = np.asarray(direction) / np.linalg.norm(direction)
        right = np.cross(up, backward)
        right /= np.linalg.norm(right)
        upward = np.cross(backward, right)
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack((right, upward, backward))
        matrix[:3, 3] = center + 3 * span * backward
        views.append({"id": name, "projection": "orthographic", "camera_to_world": matrix.tolist(),
                      "orthographic_scale": 1.65 * span, "clip_start": .01 * span,
                      "clip_end": 10 * span})
    frozen = render.validate_cameras({"schema": "simready-cameras-v1", "resolution": [resolution, resolution],
                                     "observed_camera_id": "observation", "views": views[:4],
                                     "observation": views[4]})
    if task_id is not None:
        from .observation import freeze_observation
        return freeze_observation(frozen, [low.tolist(), high.tolist()], task_id)
    return frozen


def _normalized_render(receipt):
    value = json.loads(json.dumps(receipt, allow_nan=False))
    for image in value.get("images", []):
        for key in ("file", "mask_file"):
            image[key] = Path(image[key]).name
    return value


def prepare(source, package_root, output, task_id, *, cameras=None, timeout=180, metadata=None):
    """Return a typed receipt; on any failure leave no newly qualified output."""
    receipt = {"schema": SCHEMA, "status": "unknown", "task_id": task_id}
    try:
        source, root, output = Path(source).absolute(), Path(package_root).absolute(), Path(output).absolute()
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", task_id):
            raise _Reject("safe_task_id_required")
        if not math.isfinite(float(timeout)) or not 0 < float(timeout) <= 3600:
            raise _Reject("bounded_timeout_required")
        if output.exists() or output.is_symlink():
            raise _Reject("fresh_output_required")
        if _forbidden(root) or _forbidden(output):
            raise _Reject("legacy_or_result_content_forbidden")
        if not source.is_relative_to(root) or source.resolve() != source:
            raise _Reject("qualified_entrypoint_required")
        entrypoint = source.relative_to(root).as_posix()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise _Reject("output_must_be_separate_from_package")
        files = _scan(root)
        for name in ("eligibility.json", "observation-mask.png", "observation-render-receipt.json"):
            if root / name in files:
                raise _Reject("reserved_reference_filename", path=name)
        package = inspect_package(root, entrypoint=entrypoint, allow_nested_entrypoint=True,
                                  trusted_reference=True)
        if not package["package_ok"] or package["infrastructure_errors"]:
            raise _Reject("package_closure_unavailable", package_report=package)
        _closure(source, root, files)
        source_files = [_row(path, root) for path in files]
        metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        if not isinstance(metadata, dict):
            raise _Reject("metadata_object_required")
        if output.parent.resolve() != output.parent:
            raise _Reject("regular_output_parent_required")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".simready-prepare-", dir=output.parent) as scratch:
            staging = Path(scratch) / "task"
            reference, comparison, public = (staging / role for role in ("reference", "comparison", "public"))
            for directory in (reference, comparison, public):
                directory.mkdir(parents=True)
            for row in source_files:
                destination = reference / row["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / row["path"], destination)
                if _hash(destination) != row["sha256"]:
                    raise _Reject("source_changed_while_copying")
            observation = observe(reference / entrypoint)
            geometry = observation["eligibility"]
            if not len(observation["triangles"]) or geometry["supported_prims"] != geometry["visible_prims"] or any(
                item.get("geometry") for item in geometry["unsupported"]):
                raise _Reject("unsupported_reference_geometry", support=geometry)
            dimensions = metadata.get("nominal_dimensions_m")
            if dimensions is None:
                points = np.asarray(observation["triangles"], float).reshape(-1, 3)
                dimensions = (points.max(axis=0) - points.min(axis=0)).tolist()
            if (not isinstance(dimensions, list) or len(dimensions) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or not 1e-9 <= value <= 1e6 for value in dimensions)):
                raise _Reject("public_nominal_dimensions_required")
            metadata["nominal_dimensions_m"] = dimensions
            instruction = public_instruction(dimensions)
            frozen = render.validate_cameras(cameras) if cameras is not None else freeze_cameras(observation, task_id=task_id)
            separate_observation = frozen.get("observation")
            comparison_ids = {view["id"] for view in frozen["views"]}
            if (not isinstance(separate_observation, dict)
                    or frozen["observed_camera_id"] != separate_observation.get("id")
                    or separate_observation["id"] in comparison_ids):
                raise _Reject("distinct_observation_camera_required")
            artifacts = {view["id"] + suffix for view in frozen["views"]
                         for suffix in (".png", "-mask.png")}
            if len(artifacts) != 8:
                raise _Reject("camera_artifact_collision")
            _write(comparison / "cameras.json", frozen)
            batch = render.render_asset(reference / entrypoint, frozen, Path(scratch) / "render",
                                        timeout=float(timeout), role="all", package_root=reference)
            if batch.get("status") != "measured":
                raise _Reject(batch.get("reason", "render_unavailable"), support=batch.get("support", {}),
                              render_receipt=_normalized_render(batch))
            images = batch.get("images", [])
            by_camera = {image["camera_id"]: image for image in images}
            if len(images) != 5 or set(by_camera) != comparison_ids | {separate_observation["id"]}:
                raise _Reject("five_camera_render_contract_mismatch")
            rendered = {**batch, "role": "comparison", "render_batch_role": "all",
                        "images": [by_camera[view["id"]] for view in frozen["views"]]}
            observed = {**batch, "role": "observation", "render_batch_role": "all",
                        "images": [by_camera[separate_observation["id"]]]}
            for image in rendered["images"]:
                name = image["camera_id"]
                for key, suffix, mode in (("file", ".png", "RGB"), ("mask_file", "-mask.png", "L")):
                    target = comparison / (name + suffix)
                    shutil.copyfile(image[key], target)
                    with Image.open(target) as decoded:
                        if decoded.mode != mode or list(decoded.size) != frozen["resolution"]:
                            raise _Reject("render_artifact_contract_mismatch")
                        if mode == "L" and not np.any(np.asarray(decoded)):
                            raise _Reject("empty_reference_foreground", camera_id=name)
            _write(comparison / "render-receipt.json", _normalized_render(rendered))
            observation_image = observed["images"][0]
            shutil.copyfile(observation_image["file"], public / "observation.png")
            shutil.copyfile(observation_image["mask_file"], reference / "observation-mask.png")
            for artifact, mode in ((public / "observation.png", "RGB"), (reference / "observation-mask.png", "L")):
                with Image.open(artifact) as decoded:
                    if decoded.mode != mode or list(decoded.size) != frozen["resolution"]:
                        raise _Reject("observation_artifact_contract_mismatch")
                    if mode == "L" and not np.any(np.asarray(decoded)):
                        raise _Reject("empty_observation_foreground")
            observation_receipt = _normalized_render(observed)
            observation_receipt["images"][0].update(file="../public/observation.png", mask_file="observation-mask.png")
            _write(reference / "observation-render-receipt.json", observation_receipt)
            (public / "instruction.md").write_text(instruction)
            cfg = correspondence.settings({"enabled": True})
            profile = correspondence.reference_profile(observation, cfg)
            fixture_files = [_row(path, staging) for directory in (reference, comparison)
                             for path in sorted(directory.rglob("*")) if path.is_file()]
            eligibility = {"schema": "simready-reference-eligibility-v1", "files": fixture_files,
                           "profile": profile, "profile_sha256": correspondence.digest(profile),
                           "render_recipe_sha256": rendered["recipe_sha256"],
                           "render_camera_sha256": rendered["camera_sha256"],
                           "render_bridge_code_sha256": rendered["bridge_code_sha256"],
                           "renderer_binary_sha256": rendered["renderer_binary_sha256"]}
            _write(reference / "eligibility.json", eligibility)
            config = {"schema": "simready-evaluator-config-v1", "task_id": task_id,
                      "entrypoint": "asset.usd", "reference": "reference/" + entrypoint,
                      "cameras": "comparison/cameras.json", "eligibility": "reference/eligibility.json",
                      "fidelity_settings": cfg, "final_seconds": 600}
            _write(staging / "evaluator.json", config)
            delivered = []
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    row = _row(path, staging)
                    row["role"] = path.relative_to(staging).parts[0] if path.parent != staging else "configuration"
                    delivered.append(row)
            inventory = {"schema": INVENTORY_SCHEMA, "task_id": task_id, "metadata": metadata,
                         "source_entrypoint": entrypoint, "source_files": source_files,
                         "files": delivered,
                         "self_hash_policy": "asset-inventory.json excluded; receipt records its SHA-256"}
            _write(staging / "asset-inventory.json", inventory)
            if [_row(path, root) for path in _scan(root)] != source_files:
                raise _Reject("source_changed_during_preparation")
            if output.exists() or output.is_symlink():
                raise _Reject("fresh_output_required")
            staging.rename(output)
        receipt.update(status="prepared", output=str(output), reference=config["reference"],
                       cameras=config["cameras"], eligibility=config["eligibility"], config="evaluator.json",
                       asset_inventory_sha256=_hash(output / "asset-inventory.json"),
                       profile_sha256=eligibility["profile_sha256"],
                       render_recipe_sha256=rendered["recipe_sha256"], render_camera_sha256=rendered["camera_sha256"],
                       comparison_rgb_count=4, public_observation_count=1, mask_count=5,
                       comparison_mask_count=4, observation_mask_count=1)
    except _Reject as exc:
        receipt.update(reason=exc.reason, **exc.evidence)
    except (OSError, ValueError, RuntimeError, TypeError, IndexError, KeyError, MemoryError) as exc:
        receipt.update(reason="preparation_unavailable", error_type=type(exc).__name__, error=str(exc))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--cameras")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--metadata", help="Optional local JSON object with source/license provenance")
    args = parser.parse_args(argv)
    try:
        metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else None
        receipt = prepare(args.source, args.package_root, args.output, args.task_id,
                          cameras=args.cameras, timeout=args.timeout, metadata=metadata)
    except (OSError, ValueError, TypeError) as exc:
        receipt = {"schema": SCHEMA, "status": "unknown", "task_id": args.task_id,
                   "reason": "invalid_preparation_input", "error": str(exc)}
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0 if receipt["status"] == "prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
