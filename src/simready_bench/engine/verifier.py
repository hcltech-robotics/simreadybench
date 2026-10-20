"""Harbor verifier client and the NVIDIA Robotics-Prop scoring contract."""

from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import tempfile

PROTOCOL_VERSION = "usd-reconstruction/v1"
WIRE_SCHEMA = "simready-verifier/v1"
POLICY_ID = "NVIDIA-Robotics-Prop-4.0.0-mandatory-95-plus-5-v1"
PROFILE_FILE = Path(__file__).with_name("nvidia_profile.json")
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def digest(v):
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def pinned_profile():
    return json.loads(PROFILE_FILE.read_text())


def scoring_policy(*, excluded_rules=None):
    p = pinned_profile()
    codes = [r["code"] for r in p["mandatory_requirements"]]
    if excluded_rules is None:
        excluded_rules = []
    if excluded_rules:
        raise ValueError(
            "Every pinned mandatory NVIDIA requirement must be scored; filtered profiles are not supported"
        )
    if (
        not isinstance(excluded_rules, list)
        or len(set(excluded_rules)) != len(excluded_rules)
        or (not set(excluded_rules) <= set(codes))
    ):
        raise ValueError("Explicit unique mandatory exclusions required")
    return {
        "schema": "simready-validator-scoring-policy-v1",
        "policy_id": POLICY_ID,
        "profile": {"id": p["profile_id"], "version": p["profile_version"]},
        "profile_inventory_sha256": digest(p),
        "scoring_unit": "one_mandatory_requirement_ID",
        "base_credit_percent": 95,
        "all_pass_bonus_percent": 5,
        "applicable_requirement_ids": [c for c in codes if c not in excluded_rules],
        "report_only_mandatory_ids": sorted(excluded_rules),
        "optional_requirements": "report_only",
        "applicability": "fixed_profile_and_task_before_candidate",
        "unknown": "null_score_fixed_denominator",
        "scope": "filtered_mandatory_profile"
        if excluded_rules
        else "all_mandatory_profile",
        "process_shaping_percent": 0,
        "stop_bonus_percent": 0,
        "early_stop_penalty_percent": 0,
    }


def validate_policy(p):
    if not isinstance(p, dict) or p != scoring_policy(
        excluded_rules=p.get("report_only_mandatory_ids")
    ):
        raise ValueError("Exact pinned versioned mandatory scoring policy required")
    if not p["applicable_requirement_ids"]:
        raise ValueError("Nonempty fixed mandatory scoring scope required")
    return deepcopy(p)


def validate_wire(wire):
    keys = {
        "protocol",
        "schema_version",
        "episode_id",
        "snapshot_id",
        "snapshot_sha256",
        "outcome",
        "reward",
        "simready",
        "acceptance",
        "readiness",
        "errors",
    }
    if (
        not isinstance(wire, dict)
        or set(wire) != keys
        or wire["schema_version"] != WIRE_SCHEMA
        or (wire["protocol"] != PROTOCOL_VERSION)
    ):
        raise ValueError("Exact versioned validator wire required")
    score = wire["simready"]
    rows = score.get("checks", [])
    roster = pinned_profile()["mandatory_requirements"]
    if len(rows) != len(roster) or [r.get("requirement_id") for r in rows] != [
        r["code"] for r in roster
    ]:
        raise ValueError("Exact pinned mandatory ID roster required")
    excluded = [r["requirement_id"] for r in rows if r.get("scoring") is False]
    policy = scoring_policy(excluded_rules=excluded)
    if (
        score.get("policy_id") != POLICY_ID
        or score.get("policy_sha256") != digest(policy)
        or score.get("scope") != policy["scope"]
    ):
        raise ValueError("Pinned public scoring scope required")
    active = [r for r in rows if r.get("scoring") is True]
    if len(active) != len(policy["applicable_requirement_ids"]) or any(
        (r.get("status") not in ("pass", "fail", "unknown", "invalid") for r in rows)
    ):
        raise ValueError("Typed mandatory statuses required")
    n = len(active)
    passed = sum((r["status"] == "pass" for r in active))
    failed = sum((r["status"] in ("fail", "invalid") for r in active))
    unknown = sum((r["status"] == "unknown" for r in active))
    if (
        score.get("applicable_count"),
        score.get("passed_count"),
        score.get("failed_count"),
        score.get("unknown_count"),
    ) != (n, passed, failed, unknown):
        raise ValueError("Requirement accounting inconsistent")
    expected = None if unknown else 95 * passed / n + (5 if passed == n else 0)
    bonus = None if unknown else 5 if passed == n else 0
    if (
        score.get("completion_bonus") != bonus
        or score.get("score_percent") != expected
        or wire["reward"] != expected
    ):
        raise ValueError("95/5 reward inconsistent")
    if wire["acceptance"].get("accepted") != (passed == n and (not unknown)):
        raise ValueError("Completion must follow scored mandatory outcomes")
    if score.get("official_conformance", {}).get("passed") is True and (
        expected != 100 or any((r["status"] != "pass" for r in rows))
    ):
        raise ValueError("Official pass requires all unfiltered mandatory outcomes")
    if "optional_score" in score:
        from simready_bench.engine.scoring import validate_optional_score

        validate_optional_score(score["optional_score"])
    if type(wire["reward"]) is bool or (
        wire["reward"] is not None
        and (
            not isinstance(wire["reward"], (int, float))
            or not math.isfinite(wire["reward"])
            or (not 0 <= wire["reward"] <= 100)
        )
    ):
        raise ValueError("Finite 0-100 reward required")
    h = wire["snapshot_sha256"]
    if h is not None and (
        not isinstance(h, str) or not re.fullmatch("[a-f0-9]{64}", h)
    ):
        raise ValueError("Frozen snapshot hash required")
    json.dumps(wire, allow_nan=False)
    return wire


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_path)


def _token(path):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError("Private verifier token must not use symlinks")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 63
            or info.st_uid not in (0, 1000, os.getuid())
            or (info.st_size > 512)
        ):
            raise ValueError("Private verifier token permissions invalid")
        value = stream.read(513).decode("ascii").strip()
    if not re.fullmatch("[A-Za-z0-9_-]{32,512}", value):
        raise ValueError("Private verifier token invalid")
    return value


def _write(path, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".verifier-", delete=False
    ) as stream:
        os.chmod(stream.name, 384)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def validate_report(value):
    return validate_wire(value)


def write_verifier_output(report, output):
    report = validate_wire(report)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("reward.json", "reward.txt"):
        (output / name).unlink(missing_ok=True)
    _write(output / "report.json", report)
    if report["reward"] is None:
        return 1
    _write(output / "reward.json", {"simready_reward": float(report["reward"])})
    return 0


def run_verifier(socket_path, output_dir, *, timeout=590):
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or (not 0 < timeout <= 590)
    ):
        raise ValueError("Bounded timeout required")
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != output:
        raise ValueError("No symlink output")
    for name in ("reward.json", "reward.txt"):
        (output / name).unlink(missing_ok=True)
    connection = None
    try:
        path = Path(socket_path)
        token = _token(path.parent / "token")
        connection = UnixConnection(path, timeout)
        connection.request(
            "POST",
            "/internal/final",
            body=b"{}",
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Trusted final service unavailable")
        length = response.getheader("Content-Length")
        if length is not None and (
            not length.isdigit() or int(length) > MAX_RESPONSE_BYTES
        ):
            raise ValueError("Bounded response required")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Bounded response required")
        report = validate_report(json.loads(raw))
    except Exception:
        _write(
            output / "report.json",
            {
                "outcome": "infrastructure_error",
                "reward": None,
                "errors": ["verifier_transport_or_report_error"],
            },
        )
        return 1
    finally:
        if connection is not None:
            connection.close()
    return write_verifier_output(report, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--socket", type=Path, default=Path("/run/simready-verifier/service.sock")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    raise SystemExit(run_verifier(args.socket, args.output_dir))


if __name__ == "__main__":
    main()
