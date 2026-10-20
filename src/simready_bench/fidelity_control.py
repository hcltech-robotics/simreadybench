"""Bounded isolated CPU USD analysis, separate from NVIDIA reward and illustrations."""

import json, os, re, subprocess, time, uuid
from pathlib import Path
from simready_bench.fidelity import settings


def _run(condition, cfg, runtime, root, *, operation, snapshot=None, profile=None):
    selected = settings(condition["settings"].get("fidelity", {}))
    if not selected["enabled"]:
        return {"status": "disabled", "aggregate_score_percent": None}
    image = runtime.get("fidelity_image")
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Immutable CPU fidelity image required")
    binding = cfg["validator_reference"]
    package = Path(runtime["allowed_reference_root"]) / binding["package"]
    source = package / binding["dependency_manifest"]["entrypoint"]
    output = Path(root) / "private" / ("fidelity-" + operation)
    output.mkdir(mode=0o700, parents=True)
    for path in (package, source, Path(runtime["source_root"]), output):
        if path.resolve() != path or not path.exists():
            raise ValueError("Regular confined CPU analysis paths required")
    request = {
        "operation": operation,
        "reference": "/reference/" + str(source.relative_to(package)),
        "reference_sha256": binding["dependency_manifest"]["source_sha256"],
        "settings": selected,
    }
    mounts = [
        (package, "/reference", True),
        (Path(runtime["source_root"]), "/product", True),
        (output, "/analysis", False),
    ]
    if operation == "compare":
        if not profile or profile.get("status") != "frozen":
            raise ValueError("Pre-authoring reference eligibility required")
        request.update(
            profile_sha256=profile["profile_sha256"],
            candidate_invalid=not snapshot.valid,
        )
        if snapshot.valid:
            candidate = Path(snapshot.path)
            if candidate.resolve() != candidate or not candidate.is_dir():
                raise ValueError("Confined frozen candidate required")
            mounts.append((candidate, "/candidate", True))
            request["candidate"] = "/candidate/" + snapshot.entrypoint
    (output / "request.json").write_text(json.dumps(request, sort_keys=True) + "\n")
    name = "simready-fidelity-" + uuid.uuid4().hex[:12]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--pull=never",
        "--network",
        "none",
        "--read-only",
        "--user",
        str(os.getuid()) + ":" + str(os.getgid()),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=268435456,mode=1777",
    ]
    for host, destination, readonly in mounts:
        command += [
            "--mount",
            "type=bind,src="
            + str(host)
            + ",dst="
            + destination
            + (",readonly" if readonly else ""),
        ]
    command += [
        "--env",
        "PYTHONPATH=/product",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint",
        "/opt/simready/venv/bin/python",
        image,
        "-B",
        "-m",
        "simready_bench.fidelity_worker",
        "/analysis/request.json",
    ]
    started = time.monotonic()
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=selected["seconds"]
        )
        (output / "worker.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(
                "CPU source comparison failed; original log retained privately"
            )
        result = json.loads((output / "receipt.json").read_text())
        result.update(worker_image=image, seconds=time.monotonic() - started)
        return result
    finally:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def prepare_reference(condition, cfg, runtime, root):
    try:
        return _run(condition, cfg, runtime, root, operation="reference")
    except Exception as exc:
        return {
            "status": "unknown",
            "error_type": type(exc).__name__,
            "aggregate_score_percent": None,
        }


def evaluate_final(condition, cfg, runtime, snapshot, root, profile):
    try:
        return _run(
            condition,
            cfg,
            runtime,
            root,
            operation="compare",
            snapshot=snapshot,
            profile=profile,
        )
    except Exception as exc:
        return {
            "status": "unknown",
            "error_type": type(exc).__name__,
            "aggregate_score_percent": None,
            "reference_profile": profile,
            "reason": "CPU comparison unavailable; no candidate zero inferred from infrastructure failure",
        }
