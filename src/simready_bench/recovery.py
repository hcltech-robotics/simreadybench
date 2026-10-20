"""Retain reconstruction attempts and bound automatic recovery."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from uuid import uuid4


class RecoveryBudgetExhausted(ValueError):
    """Remaining work cannot fit within the original project allowance."""


class RecoveryLimitReached(RuntimeError):
    """An incomplete reconstruction has exhausted its automatic retries."""


def validate_recovery(max_retries, retry_delay_seconds, maximum_retry_delay_seconds):
    if type(max_retries) is not int or not 0 <= max_retries <= 10:
        raise ValueError("Automatic retries must be an integer between 0 and 10")
    for value in (retry_delay_seconds, maximum_retry_delay_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Recovery delays must be finite nonnegative seconds")
    if retry_delay_seconds > maximum_retry_delay_seconds:
        raise ValueError("The maximum recovery delay must cover the initial delay")


def _save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    if path.is_symlink():
        raise ValueError("Recovery state must remain a regular file")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def recovery_state(
    output, *, max_retries, retry_delay_seconds, maximum_retry_delay_seconds
):
    from simready_bench.resume import _read

    output = Path(output)
    path = output / "recovery-state.json"
    policy = {
        "max_retries": max_retries,
        "retry_delay_seconds": retry_delay_seconds,
        "maximum_retry_delay_seconds": maximum_retry_delay_seconds,
    }
    binding = hashlib.sha256((output / "run.json").read_bytes()).hexdigest()
    if path.exists():
        state = _read(path)
        if (
            state.get("schema") != "simready-bench-recovery-v1"
            or state.get("run_sha256") != binding
            or state.get("policy") != policy
        ):
            raise ValueError("Recovery must retain its saved run and retry policy")
    else:
        state = {
            "schema": "simready-bench-recovery-v1",
            "run_sha256": binding,
            "policy": policy,
            "status": "ready",
            "retry_waves": 0,
            "stalled_waves": 0,
            "dispatch_waves": 0,
            "condition_attempts": {},
            "not_before": None,
            "events": [],
        }
    state.setdefault("stalled_waves", 0)
    state.setdefault("dispatch_waves", 0)
    return path, state


def record_state(path, state, status, **values):
    state.update(status=status, **values)
    state["events"].append({"status": status, "at": time.time(), **values})
    _save(path, state)


def attempted_conditions(output):
    from simready_bench.resume import _batches, _completed, _reports, _unrun

    attempts, completed = {}, set()
    for batch in _batches(Path(output)):
        for attempt, _, report in _reports(batch):
            for row in report["results"]:
                identity = row["condition_id"]
                if _completed(attempt, row):
                    completed.add(identity)
                if not _unrun(row):
                    attempts.setdefault(identity, set()).add(str(batch))
    return {identity: len(owners) for identity, owners in attempts.items()}, completed


def frozen_wheel(output, supplied=None):
    """Reuse the evaluator package bound to the original Daytona execution."""
    from simready_bench.resume import _batches, _read

    output = Path(output).absolute()
    batches = _batches(output)
    hashes, admitted = set(), False

    def remember(digest):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("A saved evaluator wheel identity is invalid")
        hashes.add(digest)

    for batch in batches:
        delivery = batch / "daytona-delivery.json"
        if not delivery.is_file():
            continue
        record = _read(delivery)
        admitted = admitted or bool(record.get("parent_allocation"))
        digest = record.get("wheel_sha256")
        if digest is not None:
            remember(digest)
        elif record.get("parent_allocation"):
            raise ValueError(
                "An admitted execution requires its original evaluator hash"
            )
    for path in sorted(output.glob("parallel-plan-*.json")):
        digest = _read(path).get("wheel_sha256")
        if digest is not None:
            remember(digest)
    if len(hashes) > 1:
        raise ValueError("Saved executions have conflicting evaluator wheel identities")
    expected = next(iter(hashes), None)

    def checked(path):
        path = Path(path).absolute()
        if (
            path.resolve() != path
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.suffix != ".whl"
        ):
            raise ValueError("The frozen evaluator wheel must remain a regular file")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    if supplied is not None:
        path, actual = checked(supplied)
        if expected is not None and actual != expected:
            raise ValueError("The evaluator wheel differs from the original execution")
        return path
    candidates = []
    for batch in batches:
        for package in [
            batch / ".executor-package",
            *sorted(batch.glob(".parallel-package-*")),
        ]:
            if package.is_dir():
                candidates.extend(
                    checked(path) for path in sorted(package.glob("*.whl"))
                )
    if expected is not None:
        matching = [path for path, actual in candidates if actual == expected]
        if not matching:
            raise ValueError("The original frozen evaluator wheel is missing")
        return matching[0]
    if admitted:
        raise ValueError("An admitted execution requires its original evaluator hash")
    if len({digest for _, digest in candidates}) > 1:
        raise ValueError(
            "Retained evaluator packages have conflicting wheel identities"
        )
    return candidates[0][0] if candidates else None


def salvage_draft(batch, attempt):
    """Derive a report from a fully retained terminal export without rewriting it."""
    from simready_bench.resume import (
        _completed,
        _digest,
        _file_receipt,
        _read,
        _tree,
        condition_plan,
    )
    from simready_bench.reporting import (
        read_generated_report,
        render_html,
        write_scores,
    )

    batch, attempt = Path(batch), Path(attempt)
    delivery = batch / "daytona-delivery.json"
    draft = attempt / ".report-export"
    source = draft / "results.json"
    if not delivery.is_file() or not source.is_file():
        return None
    receipt = _read(delivery)
    terminal = batch / "daytona-executor/controller-status.json"
    if (
        receipt.get("collection") != "completed"
        or not receipt.get("cleanup_confirmed")
        or not terminal.is_file()
        or _read(terminal).get("status") not in ("failed", "completed", "incomplete")
    ):
        return None
    if draft.resolve() != draft or not draft.is_relative_to(batch):
        raise ValueError("A retained report draft must remain within its execution")
    raw = _read(source)
    report = read_generated_report(source)
    if (
        raw.get("schema") != "simready-bench-portable-report-v1"
        or not isinstance(raw.get("configuration_sha256"), str)
        or len(raw["configuration_sha256"]) != 64
    ):
        raise ValueError("A retained report draft requires its configuration identity")
    expected = condition_plan(_read(batch / "experiment.json"))
    seen = set()
    for row in report["results"]:
        identity = row.get("condition_id")
        if (
            identity in seen
            or identity not in expected
            or any(
                row.get(key) is not None and row[key] != value
                for key, value in expected[identity].items()
            )
        ):
            raise ValueError(
                "A retained report draft changes a reconstruction identity"
            )
        seen.add(identity)
        for item in row.get("images", []) + row.get("traces", []):
            if (
                not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
                or not _file_receipt(
                    draft, {**item, "path": "evidence/" + item["path"]}
                )
            ):
                raise ValueError("A retained report draft has changed evidence")
        if row.get("status") == "completed" and not _completed(draft, row):
            raise ValueError("A completed report draft requires its frozen artefacts")
    files = _tree(draft)
    if "recovery.json" in files:
        raise ValueError("A retained report draft cannot contain a recovery receipt")
    if (
        len(files) > 10000
        or sum((draft / name).stat().st_size for name in files) > 1024**3
    ):
        raise ValueError("A retained report draft exceeds the export limit")
    proof = {
        "schema": "simready-bench-recovered-report-v1",
        "source": str(source.relative_to(batch)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "files": files,
        "conditions": sorted(seen),
    }
    recovered = batch / "recovered-reports"
    recovered.mkdir(mode=0o700, exist_ok=True)
    if recovered.resolve() != recovered:
        raise ValueError("Recovered reports must remain in their owned execution")
    target = recovered / (attempt.name + "-" + _digest(proof)[:16])
    if target.exists():
        if _read(target / "recovery.json") != proof:
            raise ValueError("A recovered report receipt has changed")
        retained = _tree(target)
        for name, digest in files.items():
            if retained.get(name) != digest:
                raise ValueError("A recovered report artefact has changed")
    else:
        incoming = recovered / (".incoming-" + uuid4().hex)
        try:
            shutil.copytree(draft, incoming)
            if _tree(incoming) != files or _tree(draft) != files:
                raise ValueError("A retained report draft changed during recovery")
            if not (incoming / "report.html").is_file():
                (incoming / "report.html").write_text(
                    render_html(report), encoding="utf-8"
                )
            if not (incoming / "scores.csv").is_file():
                write_scores(incoming / "scores.csv", report["results"])
            _save(incoming / "recovery.json", proof)
            incoming.rename(target)
        except BaseException:
            if incoming.exists() and incoming.resolve() == incoming:
                shutil.rmtree(incoming)
            raise
    return target
