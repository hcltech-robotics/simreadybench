"""Offline evaluator entry point with frozen inputs and bounded native workers.

Reference eligibility is exporter-owned and must already exist. Every worker is a
new process group with CPU, address-space and file-size limits. The container
provides filesystem/network isolation; this module never downloads fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time

from . import contract

SCHEMA = "simready-evaluator-config-v1"
ELIGIBILITY_SCHEMA = "simready-reference-eligibility-v1"
MAX_FILES = 10_000
MAX_BYTES = 256 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hash(path, deadline=None):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Final deadline exceeded while hashing inputs")
            result.update(chunk)
    return result.hexdigest()


def _json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON input exceeds the evaluator limit")
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("Nonfinite JSON number")))


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError("Nonempty bounded relative path required")
    result = PurePosixPath(value)
    if result.is_absolute() or ".." in result.parts or any(c in value for c in "\\:$~\0"):
        raise ValueError("Safe relative fixture path required")
    if result.as_posix() != value or value == ".":
        raise ValueError("Canonical relative fixture path required")
    return result


def _regular(root, relative):
    relative = _relative(relative)
    current = Path(root)
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Fixture links are forbidden")
    info = current.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Regular unlinked fixture files required")
    return current


def _configuration(value):
    from .correspondence import settings
    value = _json(value) if isinstance(value, (str, Path)) else value
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Expected " + SCHEMA)
    result = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(result.get("task_id"), str) or not result["task_id"]:
        raise ValueError("Task identity required")
    for key in ("entrypoint", "reference", "cameras", "eligibility"):
        _relative(result.get(key))
    if len(PurePosixPath(result["entrypoint"]).parts) != 1:
        raise ValueError("A top-level candidate entrypoint is required")
    for key, prefix in (("reference", "reference"), ("eligibility", "reference"),
                        ("cameras", "comparison")):
        if PurePosixPath(result[key]).parts[0] != prefix:
            raise ValueError("Fixture role path mismatch: " + key)
    result["fidelity_settings"] = settings(result.get("fidelity_settings", {"enabled": True}))
    seconds = result.get("final_seconds", 600)
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 1_000_000_000:
        raise ValueError("Final deadline must be in (0,1000000000] seconds")
    result["final_seconds"] = seconds
    return result


def freeze_candidate(source, destination, *, entrypoint="asset.usd", deadline=None):
    """Copy only the declared subtree using no-follow file descriptors and bounds.

    Hashes and read-only permissions are checked again after native processing.
    Unexpected filesystem failures remain infrastructure failures.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    result = {"status": "unknown", "proven_invalid": False, "files": [], "directories": []}
    total = entries = 0
    class Invalid(ValueError):
        pass
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Final deadline exceeded while freezing")
    try:
        _relative(entrypoint)
        if len(PurePosixPath(entrypoint).parts) != 1:
            raise Invalid("invalid_entrypoint")
        if source.is_symlink() or source.resolve() != source:
            raise Invalid("symlink_not_allowed")
        if not source.exists():
            raise Invalid("missing_entrypoint")
        if not source.is_dir():
            raise Invalid("special_file_not_allowed")
        destination.mkdir(parents=True, exist_ok=False)
        def walk(descriptor, target, prefix="", depth=0):
            nonlocal entries, total
            if depth > 64:
                raise Invalid("package_too_deep")
            with os.scandir(descriptor) as scan:
                for item in scan:
                    check_deadline()
                    entries += 1
                    if entries > MAX_FILES:
                        raise Invalid("package_entry_limit_exceeded")
                    name = item.name
                    relative = prefix + name
                    try:
                        _relative(relative)
                    except ValueError:
                        raise Invalid("unsafe_path") from None
                    info = item.stat(follow_symlinks=False)
                    path = target / name
                    if stat.S_ISDIR(info.st_mode):
                        result["directories"].append(relative)
                        path.mkdir()
                        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                        try:
                            walk(fd, path, relative + "/", depth + 1)
                        finally:
                            os.close(fd)
                        path.chmod(0o555)
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        if total + info.st_size > MAX_BYTES:
                            raise Invalid("package_too_large")
                        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
                        with os.fdopen(fd, "rb") as src, path.open("xb") as dst:
                            opened = os.fstat(src.fileno())
                            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                                raise Invalid("nonregular_or_linked_file")
                            written = 0
                            digest = hashlib.sha256()
                            while chunk := src.read(1024 * 1024):
                                check_deadline()
                                written += len(chunk)
                                if total + written > MAX_BYTES:
                                    raise Invalid("package_too_large")
                                digest.update(chunk)
                                dst.write(chunk)
                            after = os.fstat(src.fileno())
                            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
                                opened.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                after.st_mtime_ns, after.st_ctime_ns) or written != opened.st_size:
                                raise OSError("Candidate changed while freezing")
                        total += written
                        path.chmod(0o444)
                        result["files"].append({"path": relative, "size_bytes": written,
                                                "sha256": digest.hexdigest()})
                    else:
                        raise Invalid("nonregular_or_linked_file")
        fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            walk(fd, destination)
        finally:
            os.close(fd)
        if entrypoint not in {row["path"] for row in result["files"]}:
            raise Invalid("missing_entrypoint")
        result["files"].sort(key=lambda row: row["path"])
        result["directories"].sort()
        destination.chmod(0o555)
        result.update(status="frozen", package_bytes=total, package_entries=entries,
                      sha256=_digest(result["files"]), entrypoint=entrypoint)
    except Invalid as exc:
        result.update(status="invalid", proven_invalid=True, reason=str(exc))
    except Exception as exc:
        result.update(reason="candidate_freeze_error", error_type=type(exc).__name__)
    return result


def verify_candidate(root, receipt):
    try:
        paths, directories = set(), set()
        for path in Path(root).rglob("*"):
            if path.is_symlink():
                return False
            relative = path.relative_to(root).as_posix()
            (directories if path.is_dir() else paths).add(relative)
        if directories != set(receipt.get("directories", [])):
            return False
        if paths != {row["path"] for row in receipt["files"]}:
            return False
        for row in receipt["files"]:
            path = _regular(root, row["path"])
            if path.stat().st_size != row["size_bytes"] or _hash(path) != row["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


RENDER_FILE_LIMIT_BYTES = 4 * 1024**3
RENDER_ADDRESS_SPACE_BYTES = 12 * 1024**3


def _worker_limits(seconds, file_limit_bytes=128 * 1024**2, address_space_bytes=8 * 1024**3):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(seconds)), max(2, math.ceil(seconds) + 1)))
    resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit_bytes, file_limit_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_process(command, log, *, timeout, file_limit_bytes=128 * 1024**2, address_space_bytes=8 * 1024**3):
    if timeout <= 0:
        return {"status": "unknown", "reason": "final_deadline_exceeded"}
    with Path(log).open("wb") as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: _worker_limits(timeout, file_limit_bytes, address_space_bytes))
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return {"status": "unknown", "reason": "worker_timeout"}
    return {"status": "complete" if code == 0 else "unknown",
            **({"reason": "worker_failed", "returncode": code} if code else {})}


def _run_worker(mode, request, scratch, deadline, *, seconds=120):
    remaining = min(seconds, deadline - time.monotonic())
    if remaining <= 0:
        return {"status": "unknown", "reason": "final_deadline_exceeded"}
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    request_file, result_file = scratch / "request.json", scratch / "result.json"
    request_file.write_text(json.dumps(request, allow_nan=False))
    try:
        status = _run_process([sys.executable, "-m", "simready_eval.cli", "_worker", mode,
                               str(request_file), str(result_file)], scratch / "worker.log", timeout=remaining,
                              file_limit_bytes=RENDER_FILE_LIMIT_BYTES if mode == "render" else 128 * 1024**2,
                              address_space_bytes=RENDER_ADDRESS_SPACE_BYTES if mode == "render" else 8 * 1024**3)
    except OSError as exc:
        return {"status": "unknown", "reason": "worker_launch_failed", "error_type": type(exc).__name__}
    if status["status"] != "complete":
        return status
    try:
        return _json(result_file)
    except (OSError, ValueError):
        return {"status": "unknown", "reason": "invalid_worker_result"}


def _fixtures(root, config, deadline=None):
    from . import render
    from .render import RECIPE, _identity, validate_cameras
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError("Regular fixture root required")
    eligibility = _json(_regular(root, config["eligibility"]))
    if eligibility.get("schema") != ELIGIBILITY_SCHEMA:
        raise ValueError("Expected " + ELIGIBILITY_SCHEMA)
    files = eligibility.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Bounded frozen fixture inventory required")
    indexed = {}
    total = 0
    for row in files:
        path = row["path"]
        _relative(path)
        if PurePosixPath(path).parts[0] not in ("reference", "comparison") or path in indexed:
            raise ValueError("Unique reference/comparison fixture files required")
        if path == config["eligibility"]:
            raise ValueError("Eligibility manifest cannot hash itself")
        if type(row.get("size_bytes")) is not int or row["size_bytes"] < 0:
            raise ValueError("Nonnegative fixture size required")
        total += row["size_bytes"]
        if total > 2 * 1024**3:
            raise ValueError("Fixture closure exceeds 2 GiB")
        artifact = _regular(root, path)
        if artifact.stat().st_size != row["size_bytes"] or _hash(artifact, deadline) != row.get("sha256"):
            raise ValueError("Frozen fixture digest mismatch: " + path)
        indexed[path] = row
    actual, entries = set(), 0
    for directory in (root / "reference", root / "comparison"):
        for path in directory.rglob("*"):
            entries += 1
            if entries > MAX_FILES + 1:
                raise ValueError("Fixture entry limit exceeded")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Final deadline exceeded while inspecting fixtures")
            if path.is_file() or path.is_symlink():
                actual.add(path.relative_to(root).as_posix())
    if actual - {config["eligibility"]} != set(indexed):
        raise ValueError("Fixture closure contains unrecorded or missing files")
    if config["reference"] not in indexed or config["cameras"] not in indexed:
        raise ValueError("Reference and cameras must be frozen in the inventory")
    cameras = validate_cameras(_json(root / config["cameras"]))
    if eligibility.get("render_recipe_sha256") != _identity(RECIPE):
        raise ValueError("Frozen reference render recipe identity mismatch")
    if eligibility.get("render_camera_sha256") != _identity(cameras):
        raise ValueError("Frozen reference render camera identity mismatch")
    bridge = _identity([render._hash(render.__file__),
                        render._hash(Path(render.__file__).with_name("render_worker.py"))])
    if eligibility.get("render_bridge_code_sha256") != bridge:
        raise ValueError("Frozen reference render bridge identity mismatch")
    binary = shutil.which("blender")
    if binary is None or eligibility.get("renderer_binary_sha256") != render._hash(binary):
        raise ValueError("Frozen reference renderer binary identity mismatch")
    images = []
    for view in cameras["views"]:
        rgb, mask = "comparison/" + view["id"] + ".png", "comparison/" + view["id"] + "-mask.png"
        if rgb not in indexed or mask not in indexed:
            raise ValueError("Four frozen RGB images and foreground masks required")
        images.append({"camera_id": view["id"], "file": str(root / rgb),
                       "mask_file": str(root / mask), "sha256": indexed[rgb]["sha256"],
                       "mask_sha256": indexed[mask]["sha256"]})
    if _digest(eligibility.get("profile")) != eligibility.get("profile_sha256"):
        raise ValueError("Frozen reference profile digest mismatch")
    return eligibility, cameras, images


def _unknown(reason):
    return contract.component(reason=reason)


def _component(row, reason="measurement_unavailable"):
    value = row.get("score_percent")
    coverage = row.get("coverage_fraction", 0)
    low = row.get("lower_bound_percent", 0)
    high = row.get("upper_bound_percent", 100)
    return contract.component(value, coverage=coverage, lower=0 if low is None else low,
        upper=100 if high is None else high, reason=row.get("reason", reason) if value is None else None,
        evidence=row)


def _invalid_scores(report):
    from .formal import score_validation
    score = score_validation(None, candidate_invalid=True)
    report["candidate_invalid"] = True
    report["scores"] = {name: contract.component(0, coverage=1, reason="invalid_candidate_package")
                        for name in report["scores"]}
    report["scores"]["formal"]["evidence"] = score
    report.setdefault("formal_report", {"outcome": "invalid_submission", "simready": score,
                                         "optional_score": score["optional_score"]})


def evaluate_submission(submission, fixtures, config, output):
    """Evaluate one declared submission; retain three independent score dimensions."""
    started = time.monotonic()
    report = {"schema": contract.REPORT_SCHEMA, "candidate_invalid": False,
              "scores": {name: _unknown("evaluation_unavailable") for name in
                         ("formal", "geometry", "appearance", "visual", "kinematic")}, "errors": []}
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = _configuration(config)
        deadline = started + config["final_seconds"]
        report.update(task_id=config["task_id"], config_sha256=_digest(config))
        eligibility, cameras, images = _fixtures(fixtures, config, deadline)
        report["reference"] = {"closure_sha256": _digest(eligibility["files"]),
                               "profile_sha256": eligibility["profile_sha256"],
                               "render_recipe_sha256": eligibility["render_recipe_sha256"],
                               "render_camera_sha256": eligibility["render_camera_sha256"]}
        with tempfile.TemporaryDirectory(prefix="simready-final-") as temporary:
            scratch = Path(temporary)
            reference_request = {"reference": str(Path(fixtures).absolute() / config["reference"]),
                                 "reference_root": str(Path(fixtures).absolute() / "reference"),
                                 "profile": eligibility["profile"], "settings": config["fidelity_settings"],
                                 "images": images, "cameras": cameras}
            checked = _run_worker("reference", reference_request, scratch / "reference", deadline)
            if checked.get("status") != "verified":
                raise ValueError("reference_eligibility_unavailable:" + checked.get("reason", "profile_mismatch"))
            candidate = freeze_candidate(submission, scratch / "candidate", entrypoint=config["entrypoint"], deadline=deadline)
            report["candidate"] = candidate
            if candidate["status"] == "invalid":
                _invalid_scores(report)
            elif candidate["status"] != "frozen":
                raise ValueError(candidate.get("reason", "candidate_freeze_error"))
            else:
                root = scratch / "candidate"
                formal = _run_worker("formal", {"root": str(root), "entrypoint": config["entrypoint"]},
                                     scratch / "formal", deadline)
                report["formal_report"] = formal
                if formal.get("outcome") == "invalid_submission":
                    _invalid_scores(report)
                else:
                    score = formal.get("simready", {})
                    count = score.get("applicable_count", 46)
                    report["scores"]["formal"] = contract.component(score.get("score_percent"),
                        coverage=(score.get("passed_count", 0) + score.get("failed_count", 0)) / count,
                        lower=score.get("score_lower_bound", 0), upper=score.get("score_upper_bound", 100),
                        evidence=score, reason=score.get("unknown_reason") or formal.get("reason"))
                    if config["fidelity_settings"]["enabled"]:
                        comparison = _run_worker("compare", {**reference_request, "candidate": str(root / config["entrypoint"])},
                            scratch / "compare", deadline, seconds=config["fidelity_settings"]["seconds"])
                        report["correspondence"] = comparison
                        parts = comparison.get("components", {})
                        report["scores"]["geometry"] = _component(parts.get("shape", {}), comparison.get("reason", "geometry_unavailable"))
                        report["scores"]["kinematic"] = _component(parts.get("kinematics", {}), comparison.get("reason", "kinematics_not_applicable"))
                        if comparison.get("alignment") is not None:
                            receipt = _run_worker("render", {"candidate": str(root / config["entrypoint"]),
                                "cameras": cameras, "alignment": comparison["alignment"],
                                "images": images, "output": str(output / "candidate-renders"),
                                "timeout": max(1, min(180, deadline - time.monotonic()))},
                                scratch / "render", deadline, seconds=180)
                            report["render"] = receipt
                            report["scores"]["appearance"] = _component(receipt.get("appearance", {}), receipt.get("reason", "render_unavailable"))
                        report["scores"]["visual"] = contract.visual_score(report["scores"]["geometry"], report["scores"]["appearance"])
                if not verify_candidate(root, candidate):
                    report["candidate_invalid"] = False
                    report["scores"] = {name: _unknown("frozen_candidate_integrity_error") for name in report["scores"]}
                    report["errors"].append({"code": "frozen_candidate_integrity_error"})
            # Recheck trusted inputs after all native parsing, before publishing credit.
            final_eligibility, _, _ = _fixtures(fixtures, config, deadline)
            if _digest(final_eligibility) != _digest(eligibility):
                raise ValueError("Frozen reference fixtures changed during evaluation")
    except Exception as exc:
        report["candidate_invalid"] = False
        report["scores"] = {name: _unknown("evaluator_infrastructure_error") for name in report["scores"]}
        report["errors"].append({"code": "evaluator_infrastructure_error", "type": type(exc).__name__, "message": str(exc)})
    report["outcome"] = "invalid_submission" if report["candidate_invalid"] else "complete" if all(
        report["scores"][name]["score_percent"] is not None for name in contract.DIMENSIONS) else "partial"
    contract.write_outputs(report, output)
    return report


def _worker(mode, request):
    from . import correspondence, formal, render
    from .package import inspect_package
    from .usd import observe
    if mode == "formal":
        return formal.evaluate(request["root"], entrypoint=request["entrypoint"])
    if mode in ("reference", "compare"):
        reference = Path(request["reference"])
        reference_root = Path(request["reference_root"])
        preflight = inspect_package(reference_root,
            entrypoint=reference.relative_to(reference_root).as_posix(), allow_nested_entrypoint=True)
        if not preflight["package_ok"] or preflight["infrastructure_errors"]:
            return {"status": "unknown", "reason": "reference_package_unavailable", "preflight": preflight}
        observation = observe(reference)
        profile = correspondence.reference_profile(observation, request["settings"])
        if profile != request["profile"]:
            return {"status": "unknown", "reason": "reference_profile_mismatch"}
        if mode == "reference":
            from PIL import Image
            for row in request["images"]:
                for key in ("file", "mask_file"):
                    with Image.open(row[key]) as image:
                        if list(image.size) != request["cameras"]["resolution"]:
                            raise ValueError("Frozen reference resolution mismatch")
                        image.load()
                        if key == "mask_file" and image.convert("L").getextrema()[1] == 0:
                            raise ValueError("Empty frozen foreground mask")
            return {"status": "verified", "profile_sha256": _digest(profile)}
        candidate = Path(request["candidate"])
        preflight = inspect_package(candidate.parent, entrypoint=candidate.name)
        if not preflight["package_ok"] or preflight["infrastructure_errors"]:
            return {"status": "unknown", "reason": "candidate_package_unavailable", "preflight": preflight}
        return correspondence.compare(observation, observe(candidate), request["settings"])
    if mode == "render":
        receipt = render.render_asset(request["candidate"], request["cameras"], request["output"],
            alignment=request["alignment"], timeout=request["timeout"])
        return {**receipt, "appearance": render.compare_views(request["images"], receipt, request["cameras"])}
    raise ValueError("Unknown worker mode")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate the declared submission offline")
    for name in ("submission", "fixtures", "config", "output"):
        evaluate.add_argument("--" + name, type=Path, required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("mode", choices=("reference", "formal", "compare", "render"))
    worker.add_argument("request", type=Path)
    worker.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        evaluate_submission(args.submission, args.fixtures, args.config, args.output)
        return 0
    try:
        result = _worker(args.mode, _json(args.request))
    except Exception as exc:
        result = {"status": "unknown", "reason": "worker_exception", "error_type": type(exc).__name__, "error": str(exc)}
    args.result.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
