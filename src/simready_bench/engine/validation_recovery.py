"""Bound final evaluation retries to the same sealed candidate and original deadline."""

import asyncio
import errno
import hashlib
import json
import os
import time

import aiohttp


def _transient(error):
    current, seen = error, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ValueError, PermissionError, FileNotFoundError, IsADirectoryError)):
            return False
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return isinstance(error, (TimeoutError, ConnectionError, aiohttp.ClientError)) or (
        isinstance(error, OSError) and error.errno in {
            None, errno.EIO, errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
            errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE,
        }
    )


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


async def evaluate_final_snapshot(evaluator, controller, snapshot, task_config, episode_id, *, deadline_at, stop_reason):
    from simready_bench.engine.harbor_services import _validation_failure

    directory = controller.state_dir / "final-validation-attempts"
    directory.mkdir(mode=0o700, exist_ok=True)
    key = hashlib.sha256((episode_id + ":" + snapshot.snapshot_id).encode()).hexdigest()
    policy_path = directory / (key + ".policy.json")
    identity = {"episode_id": episode_id, "snapshot_id": snapshot.snapshot_id,
                "snapshot_sha256": snapshot.sha256, "maximum_attempts": 3}
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        if any(policy.get(field) != value for field, value in identity.items()):
            raise ValueError("Final evaluation recovery identity changed")
        deadline_at = min(deadline_at, policy["deadline_at"])
    else:
        _write(policy_path, {**identity, "deadline_at": deadline_at})
    hold_path = directory / (key + ".hold.json")
    if hold_path.exists():
        raise ValueError("Final evaluation remains on its recorded integrity hold")
    # A completed measurement is already immutable in the real evaluator. A
    # delivery retry can read it without spending another observation allowance.
    if hasattr(evaluator, "_connect"):
        with evaluator._connect() as db:
            measured = db.execute("SELECT 1 FROM finals WHERE episode=?", (episode_id,)).fetchone()
        if measured is not None:
            return await asyncio.to_thread(evaluator.evaluate, snapshot, task_config, episode_id,
                                           deadline_at, final=True, stop_reason=stop_reason)
    used = len(list(directory.glob(key + ".*.started.json")))
    last_error = None
    while used < 3 and time.time() < deadline_at:
        used += 1
        record = {"schema": "simready-final-validation-attempt-v1", **identity,
                  "attempt": used, "deadline_at": deadline_at, "recorded_at": time.time()}
        prefix = key + "." + str(used).zfill(3)
        _write(directory / (prefix + ".started.json"), {**record, "status": "started"})
        report, failure_id, error = None, None, None
        try:
            async with asyncio.timeout(max(0.001, deadline_at - time.time())):
                report = await asyncio.to_thread(evaluator.evaluate, snapshot, task_config,
                    episode_id, deadline_at, final=True, stop_reason=stop_reason)
            failure = getattr(getattr(evaluator, "observer", None), "failures", {}).get((episode_id, snapshot.snapshot_id))
            if report.get("outcome") == "infrastructure_error" and failure is not None:
                error, failure_id = failure
        except Exception as caught:
            error = caught
            failure_id = _validation_failure(controller, error, episode_id=episode_id,
                snapshot=snapshot, phase="final_evaluation", report=report)
        transient = error is not None and _transient(error)
        _write(directory / (prefix + ".finished.json"), {**record,
            "status": "failed" if error is not None else "held" if report.get("outcome") == "infrastructure_error" else "completed",
            "failure_id": failure_id, "retryable": transient,
            "report_outcome": report.get("outcome") if isinstance(report, dict) else None})
        if error is None:
            if report.get("outcome") == "infrastructure_error":
                _write(hold_path, {**identity, "reason": "unclassified_infrastructure_evidence", "attempt": used})
            return report
        last_error = error
        if not transient:
            _write(hold_path, {**identity, "reason": "deterministic_validation_failure",
                               "failure_id": failure_id, "attempt": used})
            raise error
        delay = 0.25 * 2 ** (used - 1)
        if used >= 3 or time.time() + delay >= deadline_at:
            break
        await asyncio.sleep(delay)
    if last_error is not None:
        raise last_error
    raise TimeoutError("Final evaluation recovery allowance exhausted")
