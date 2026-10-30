"""Frozen observation delivery and the hosted 64-model-turn policy."""

import base64
import binascii
import hashlib
import json
from pathlib import Path
import re


def save_runtime(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare_runtime(config_dir, environ):
    if any(name in environ for name in ("PI_NVIDIA_TIME_LIMIT_SEC", "PI_NVIDIA_FINAL_RESERVE_SEC")):
        raise ValueError("Remove obsolete time-policy environment options; this runtime enforces 64 model turns")
    image = Path(environ.get("PI_NVIDIA_OBSERVATION_PATH", "/task/observation.png"))
    if not image.is_absolute():
        raise ValueError("Observation path must be absolute")
    expected = environ.get("PI_NVIDIA_OBSERVATION_SHA256") or None
    if expected and not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Observation hash must be a SHA-256 hex digest")
    path = Path(config_dir) / "runtime.json"
    save_runtime(path, {"model_turn_limit": 64, "final_validation_turns": 4,
        "first_candidate_target_turn": 8, "observation_path": str(image),
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
