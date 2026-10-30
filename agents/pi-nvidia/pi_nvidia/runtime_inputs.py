"""Frozen observation delivery and an explicitly labeled working clock."""

import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time


def save_runtime(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare_runtime(config_dir, environ, *, started_ns=None):
    started_ns = time.monotonic_ns() if started_ns is None else started_ns
    limit = int(environ.get("PI_NVIDIA_TIME_LIMIT_SEC", "600"))
    reserve = int(environ.get("PI_NVIDIA_FINAL_RESERVE_SEC", "90"))
    allowance = 15
    if not allowance + reserve < limit <= 600 or reserve < 1:
        raise ValueError("Keep the declared authoring limit at most 600 seconds and leave room for startup and final validation")
    image = Path(environ.get("PI_NVIDIA_OBSERVATION_PATH", "/task/observation.png"))
    if not image.is_absolute():
        raise ValueError("Observation path must be absolute")
    expected = environ.get("PI_NVIDIA_OBSERVATION_SHA256") or None
    if expected and not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Observation hash must be a SHA-256 hex digest")
    now = time.monotonic_ns()
    deadline = started_ns + (limit - allowance) * 1_000_000_000
    deadline_utc = datetime.fromtimestamp(time.time() + (deadline - now) / 1e9, timezone.utc).isoformat()
    path = Path(config_dir) / "runtime.json"
    save_runtime(path, {"configured_limit_sec": limit, "startup_allowance_sec": allowance,
        "final_reserve_sec": reserve, "first_candidate_target_sec": min(120, (limit - allowance) // 3),
        "started_monotonic_ns": str(started_ns), "working_deadline_monotonic_ns": str(deadline),
        "working_deadline_utc": deadline_utc, "deadline_basis": "conservative_adapter_clock",
        "outer_deadline_known": False, "observation_path": str(image),
        "expected_image_sha256": expected.lower() if expected else None,
        "submission_entrypoint": "/output/submission/asset.usd"})
    return path


def attach_observation(prompt, config):
    path = Path(config["observation_path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("Frozen observation must be a readable regular file")
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Frozen observation must contain the provided PNG bytes")
    digest = hashlib.sha256(data).hexdigest()
    expected = config.get("expected_image_sha256")
    if expected and digest != expected:
        raise ValueError("Frozen observation hash does not match the selected task")
    result = list(prompt)
    already = False
    for block in result:
        if block.get("type") == "image" and block.get("mimeType") == "image/png":
            try:
                already |= hashlib.sha256(base64.b64decode(block.get("data", ""), validate=True)).hexdigest() == digest
            except (binascii.Error, ValueError):
                pass
    if not already:
        result.append({"type": "image", "mimeType": "image/png", "data": base64.b64encode(data).decode("ascii")})
    return result, {"status": "verified", "source_path": str(path), "source_sha256": digest,
        "payload_sha256": digest, "bytes": len(data), "mime_type": "image/png",
        "expected_hash_verified": expected is not None,
        "delivery": "already_supplied" if already else "attached"}
