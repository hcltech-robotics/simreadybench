"""One explicit workflow shared by native Codex and hosted ACP."""

import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def bundle_identity():
    return json.loads((PACKAGE / "pins.json").read_text())


def verify_bundle(package=PACKAGE):
    package = Path(package)
    pins = json.loads((package / "pins.json").read_text())
    for relative, expected in pins["files"].items():
        if hashlib.sha256((package / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Agent bundle checksum mismatch: {relative}")
    return pins


def render_instruction(instruction, prefix="/opt/codex-text2cad"):
    verify_bundle()
    skill = (PACKAGE / "vendor/cad/SKILL.md").read_text()
    policy = (PACKAGE / "workflow.txt").read_text().replace("{{PREFIX}}", str(prefix))
    return f"{policy}\n\n<selected-cad-skill>\n{skill}\n</selected-cad-skill>\n\n<benchmark-instruction>\n{instruction}\n</benchmark-instruction>"


def verify_generation(receipt_path, asset_path):
    """Verify linked files after execution; readiness smoke receipts never qualify."""
    receipt_path, asset_path = Path(receipt_path), Path(asset_path).resolve()
    if not receipt_path.is_file():
        return {"status": "unverified", "reason": "missing_generation_receipt"}
    try:
        data = json.loads(receipt_path.read_text())
        if data["kind"] != "cad_generation" or data["bundle"] != bundle_identity():
            raise ValueError("wrong generation kind or bundle")
        if Path(data["asset_path"]).resolve() != asset_path:
            raise ValueError("receipt does not describe submitted asset")
        files = [(asset_path, data["asset_sha256"]), (Path(data["source_path"]), data["source_sha256"])]
        if Path(data["source_path"]).name == "smoke.py" or not data["parts"]:
            raise ValueError("preflight is not reconstruction evidence")
        files.extend((Path(p["step_path"]), p["step_sha256"]) for p in data["parts"])
        for path, expected in files:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"changed generation artifact: {path.name}")
        return {"status": "artifact_chain_verified", "asset_sha256": data["asset_sha256"],
                "trajectory_audit_required": True}
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {"status": "unverified", "reason": str(exc)}
