"""Snapshot integrity and stopped-actor submission checks."""

from dataclasses import asdict
from pathlib import Path
import hashlib, json, os, re, stat
from simready_bench.engine.contracts import SnapshotFile


def canonical_sha256(record):
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


PACKAGE_INVALID_CODES = frozenset(
    {
        "missing_entrypoint",
        "candidate_unreadable_or_unsafe",
        "symlink_not_allowed",
        "hardlink_not_allowed",
        "special_file_not_allowed",
        "unsafe_path",
        "package_too_large",
        "package_too_deep",
        "package_entry_limit_exceeded",
    }
)
SNAPSHOT_FIELDS = {
    "snapshot_id",
    "episode_id",
    "sha256",
    "path",
    "files",
    "size_bytes",
    "entrypoint",
    "error_code",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class EvaluationInfrastructureError(RuntimeError):
    pass


def verify_frozen_snapshot(snapshot):
    """Independently bind immutable copied bytes before and after evaluation."""
    if not snapshot.valid:
        return False
    root = Path(snapshot.path)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationInfrastructureError("snapshot_storage_changed")
    actual = []
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode):
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise EvaluationInfrastructureError("snapshot_storage_changed")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with os.fdopen(fd, "rb", closefd=False) as stream:
                h = hashlib.file_digest(stream, "sha256").hexdigest()
        finally:
            os.close(fd)
        actual.append(SnapshotFile(path.relative_to(root).as_posix(), st.st_size, h))
    expected = list(snapshot.files)
    if actual != expected or sum((f.size_bytes for f in actual)) != snapshot.size_bytes:
        raise EvaluationInfrastructureError("snapshot_storage_changed")
    if (
        digest(
            {"entrypoint": snapshot.entrypoint, "files": [asdict(f) for f in actual]}
        )
        != snapshot.sha256
    ):
        raise EvaluationInfrastructureError("snapshot_manifest_changed")
    return True


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("SHA256 pin required")
    return value


def _text(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or any((ord(x) < 32 for x in value))
    ):
        raise ValueError("bounded identity required")
    return value


def _snapshot(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != SNAPSHOT_FIELDS:
        raise ValueError("exact native contracts.Snapshot fields required")
    _text(snapshot["snapshot_id"])
    _text(snapshot["episode_id"])
    if (
        snapshot["entrypoint"] != "asset.usd"
        or type(snapshot["size_bytes"]) is not int
        or snapshot["size_bytes"] < 0
    ):
        raise ValueError("native admission contract differs")
    if not isinstance(snapshot["files"], list) or len(snapshot["files"]) > 10000:
        raise ValueError("native bounded snapshot roster required")
    previous = ""
    total = 0
    for f in snapshot["files"]:
        if not isinstance(f, dict) or set(f) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("native SnapshotFile required")
        name = f["relative_path"]
        if (
            not isinstance(name, str)
            or not name
            or name <= previous
            or name.startswith("/")
            or ("\\" in name)
            or any((p in ("", "..", ".") for p in name.split("/")))
        ):
            raise ValueError("sorted safe native file roster required")
        if type(f["size_bytes"]) is not int or f["size_bytes"] < 0:
            raise ValueError("native file byte count required")
        _hash(f["sha256"])
        previous = name
        total += f["size_bytes"]
    if snapshot["sha256"] is not None:
        _hash(snapshot["sha256"])
    if snapshot["path"] is not None and (
        not isinstance(snapshot["path"], str) or not snapshot["path"].startswith("/")
    ):
        raise ValueError("native absolute frozen path required")
    if snapshot["error_code"] is not None:
        _text(snapshot["error_code"])
    valid = (
        snapshot["error_code"] is None
        and snapshot["sha256"] is not None
        and (snapshot["path"] is not None)
    )
    if snapshot["sha256"] is not None or snapshot["path"] is not None:
        if snapshot["sha256"] is None or snapshot["path"] is None:
            raise ValueError("paired native frozen path/hash required")
        if (
            total != snapshot["size_bytes"]
            or canonical_sha256(
                {"entrypoint": snapshot["entrypoint"], "files": snapshot["files"]}
            )
            != snapshot["sha256"]
        ):
            raise ValueError("native full-package manifest SHA mismatch")
    return valid


def verify_native_invalid_evidence(
    episode,
    snapshot,
    stop,
    *,
    expected_episode_id,
    expected_snapshot_id,
    expected_task_id,
    static_receipt=None,
    expected_native_lifecycle_sha256=None,
):
    """Adapt actual SQLite episode/Snapshot and source-bound native stopped-container records.

    This adapter alone emits no normalized grade or reference coverage. Missing
    stop evidence and non-candidate failures remain explicitly unproven.
    """
    for x in (expected_episode_id, expected_snapshot_id, expected_task_id):
        _text(x)
    valid = _snapshot(snapshot)
    if (
        not isinstance(episode, dict)
        or episode.get("episode_id") != expected_episode_id
        or episode.get("task_id") != expected_task_id
        or (snapshot["episode_id"] != expected_episode_id)
        or (snapshot["snapshot_id"] != expected_snapshot_id)
        or (episode.get("final_snapshot_id") != expected_snapshot_id)
    ):
        raise ValueError("controller final snapshot/case identity differs")
    result = {
        "proven_invalid": False,
        "basis": None,
        "reason": "actor_stop_unproven",
        "error_code": snapshot["error_code"],
        "native_snapshot_valid": valid,
    }
    if stop is None:
        return result
    if not isinstance(stop, dict) or any(
        (
            stop.get(k) != v
            for k, v in {
                "episode_id": expected_episode_id,
                "snapshot_id": expected_snapshot_id,
                "snapshot_sha256": snapshot["sha256"],
                "snapshot_error_code": snapshot["error_code"],
            }.items()
        )
    ):
        raise ValueError("durable native stop/snapshot identity differs")
    state = stop.get("state", {})
    native_stop = (
        stop.get("status") == "confirmed_native_actor_stopped"
        and stop.get("actor_start_attempted") is True
        and (stop.get("producer") == "BenchmarkDockerEnvironment.stop_actor_writes")
        and (expected_native_lifecycle_sha256 is not None)
        and (stop.get("producer_source_sha256") == expected_native_lifecycle_sha256)
        and (episode.get("status") == "closed")
        and (stop.get("actor_stopped") is True)
        and isinstance(state, dict)
        and (state.get("Running") is False)
        and (state.get("Paused") is False)
        and (state.get("OOMKilled") is False)
        and (state.get("Dead") is False)
        and (state.get("Restarting") is False)
        and (state.get("Error") == "")
        and (type(state.get("ExitCode")) is int)
        and (state["ExitCode"] in (0, 137))
        and (type(stop.get("kill_return_code")) is int)
        and (stop["kill_return_code"] in (0, 1))
        and isinstance(stop.get("container_id"), str)
        and (re.fullmatch("[0-9a-f]{12,64}", stop["container_id"]) is not None)
    )
    if not native_stop:
        return result
    if not valid and snapshot["error_code"] in PACKAGE_INVALID_CODES:
        return {
            **result,
            "proven_invalid": True,
            "basis": "native_package_invalid",
            "reason": "stopped_native_final_snapshot_invalid",
        }
    result["reason"] = "candidate_invalidity_unproven"
    if static_receipt is None:
        return result
    if not valid:
        return result
    source = next(
        (
            f["sha256"]
            for f in snapshot["files"]
            if f["relative_path"] == snapshot["entrypoint"]
        ),
        None,
    )
    if (
        not isinstance(static_receipt, dict)
        or static_receipt.get("schema") != "simready-static-validation-v1"
        or source is None
        or (static_receipt.get("snapshot_sha256") != snapshot["sha256"])
        or (static_receipt.get("source_sha256") != source)
    ):
        raise ValueError("actual static invocation/source snapshot binding differs")
    if (
        static_receipt.get("profile") == {"id": "Robotics-Prop", "version": "4.0.0"}
        and static_receipt.get("execution_status") == "completed"
        and (static_receipt.get("outcome") == "invalid_submission")
        and (static_receipt.get("passed") is False)
    ):
        return {
            **result,
            "proven_invalid": True,
            "basis": "source_bound_static_invalid_submission",
            "reason": "completed_source_bound_static_invalid_submission",
        }
    return result
