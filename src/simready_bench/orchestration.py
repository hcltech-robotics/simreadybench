"""Coordinate saved Daytona executors, immutable reports and unadmitted work."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, Future, wait
from decimal import Decimal
from pathlib import Path
import threading
import time

_observer_guard = threading.Lock()
_live_observers = {}


class _Observers:
    """Host observation threads do not keep an interrupted CLI process alive."""

    def __init__(self):
        self.threads = []

    def submit(self, function, *args):
        future = Future()

        def observe():
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)

        thread = threading.Thread(target=observe, daemon=True)
        self.threads.append(thread)
        thread.start()
        return future

    def shutdown(self, *, wait):
        if wait:
            for thread in self.threads:
                thread.join()


def report_sources(batch):
    """Include sealed live reconstructions before final attempt collection."""
    from simready_bench.reporting import read_generated_report
    from simready_bench.resume import _reports

    entries = list(_reports(batch))
    seen = {str(Path(report["results"]).absolute()) for _, report, _ in entries}
    for path in sorted((batch / "reconstructions").glob("*/results.json")):
        if str(path.absolute()) not in seen:
            entries.append(
                (
                    path.parent,
                    {"directory": str(path.parent), "results": str(path)},
                    read_generated_report(path),
                )
            )
    return entries


def _completed_digest(row):
    """Bind measurements and content while allowing owned artefact relocation."""
    value = copy.deepcopy(row)
    value.pop("report_path", None)
    candidate = value.get("candidate") or {}
    for key in ("directory", "entrypoint", "manifest"):
        candidate.pop(key, None)
    for item in candidate.get("files", []):
        item.pop("path", None)
    for kind in ("images", "traces"):
        for item in value.get(kind, []):
            item.pop("path", None)
    if value.get("nvidia_validator_report_file"):
        value["nvidia_validator_report_file"].pop("path", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class ReportIndex:
    """Publish cumulative reports without coupling publication to execution."""

    def __init__(self, output, manifest, publisher=None):
        from simready_bench.resume import condition_plan

        self.output = Path(os.path.abspath(output))
        self.manifest = manifest
        self.expected = condition_plan(manifest)
        self.publisher = publisher
        self.lock = threading.RLock()
        self.entries = {}
        self.fingerprints = {}
        self.source_fingerprints = {}
        self.source_completed = {}
        self.completed_measurements = {}
        self.sources = {}
        self.completed_sources = {}
        self.closed = set()
        self.backfilling = True
        self.published_signature = None
        self.retry_after = 0
        self.result = None
        self.detached = False
        self.publication_pending = False
        self._load_index()

    def _load_index(self):
        from simready_bench.resume import _digest, _read, _unrun

        directory = self.output / ".coordinator-reports"
        path = directory / "index.json"
        if not path.exists():
            return
        saved = _read(path)
        if saved.get("schema") != "simready-bench-report-index-v1" or saved.get(
            "manifest_sha256"
        ) != _digest(self.manifest):
            raise ValueError("The retained report index changes its original matrix")
        for entry in saved["sources"]:
            snapshot = directory / entry["snapshot"]
            root = Path(entry["root"])
            if (
                snapshot.parent != directory
                or hashlib.sha256(snapshot.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError("An immutable report snapshot changed")
            value = self._check(root, snapshot, self.expected)
            key = str(snapshot)
            self.entries[key] = (root, {"directory": str(root), "results": key})
            self.fingerprints[key] = entry["sha256"]
            self.sources[key] = entry
            for row in value["results"]:
                identity = row["condition_id"]
                if row.get("status") == "completed":
                    digest = _completed_digest(row)
                    if (
                        identity in self.completed_measurements
                        and self.completed_measurements[identity] != digest
                    ):
                        raise ValueError("Retained completed measurements disagree")
                    self.completed_sources.setdefault(identity, key)
                    self.completed_measurements[identity] = digest
                if not _unrun(row):
                    self.closed.add(identity)
        self.source_completed = saved.get("source_completed", {})
        self.source_fingerprints = saved.get("source_fingerprints", {})

    def _save_index(self):
        from simready_bench.recovery import _save
        from simready_bench.resume import _digest

        _save(
            self.output / ".coordinator-reports/index.json",
            {
                "schema": "simready-bench-report-index-v1",
                "manifest_sha256": _digest(self.manifest),
                "sources": list(self.sources.values()),
                "source_completed": self.source_completed,
                "source_fingerprints": self.source_fingerprints,
            },
        )

    def _check(self, root, path, owned):
        from simready_bench.reporting import read_generated_report
        from simready_bench.resume import _completed

        if (
            root.resolve() != root
            or not root.is_dir()
            or not root.is_relative_to(self.output)
            or path.resolve() != path
            or not path.is_relative_to(self.output)
            or not path.is_file()
            or path.stat().st_nlink != 1
        ):
            raise ValueError(
                "A reconstruction report must stay within its owned output"
            )
        value = read_generated_report(path)
        seen = set()
        for row in value["results"]:
            identity = row.get("condition_id")
            if (
                identity not in owned
                or identity in seen
                or any(
                    row.get(key) is not None and row[key] != expected
                    for key, expected in self.expected[identity].items()
                )
            ):
                raise ValueError(
                    "A published reconstruction changes its owned identity"
                )
            seen.add(identity)
            if row.get("status") == "completed" and not _completed(root, row):
                raise ValueError(
                    "A completed reconstruction requires its frozen artefacts"
                )
        return value

    def add(self, batch, report, *, defer=False):
        from simready_bench.resume import _read, _unrun, condition_plan

        if self.detached:
            return self.result
        if not isinstance(report, dict) or not all(
            isinstance(report.get(key), (str, os.PathLike))
            for key in ("directory", "results")
        ):
            raise ValueError(
                "A reconstruction report requires its directory and results file"
            )
        root = Path(report["directory"]).absolute()
        path = Path(report["results"]).absolute()
        batch = Path(batch).absolute()
        if (
            not root.is_relative_to(batch)
            or not path.is_relative_to(root)
            or path.resolve() != path
            or not path.is_file()
            or path.stat().st_nlink != 1
        ):
            raise ValueError(
                "A reconstruction report must stay within its worker output"
            )
        owned = condition_plan(
            self.manifest if batch == self.output else _read(batch / "experiment.json")
        )
        raw = path.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        original = str(path)
        with self.lock:
            from simready_bench.recovery import _save

            directory = self.output / ".coordinator-reports"
            directory.mkdir(mode=0o700, exist_ok=True)
            source_id = hashlib.sha256(original.encode()).hexdigest()[:20]
            snapshot = directory / ("source-" + source_id + "-" + fingerprint + ".json")
            if not snapshot.exists():
                _save(snapshot, json.loads(raw))
            value = self._check(root, snapshot, owned)
            completed = {
                row["condition_id"]: _completed_digest(row)
                for row in value["results"]
                if row.get("status") == "completed"
            }
            if any(
                completed.get(identity) != digest
                for identity, digest in self.source_completed.get(original, {}).items()
            ):
                raise ValueError("An already retained completed reconstruction changed")
            if any(
                identity in self.completed_measurements
                and self.completed_measurements[identity] != digest
                for identity, digest in completed.items()
            ):
                raise ValueError("An already retained completed reconstruction changed")
            if self.source_fingerprints.get(original) == fingerprint:
                return self.result
            key = str(snapshot)
            self.entries[key] = (root, {**report, "results": key})
            self.fingerprints[key] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            self.sources[key] = {
                "snapshot": snapshot.name,
                "root": str(root),
                "sha256": self.fingerprints[key],
            }
            self.source_fingerprints[original] = fingerprint
            self.source_completed[original] = completed
            self.completed_measurements.update(completed)
            for row in value["results"]:
                identity = row["condition_id"]
                if row.get("status") == "completed":
                    self.completed_sources.setdefault(identity, key)
                if not _unrun(row):
                    self.closed.add(identity)
            self._save_index()
            if not defer and not self.backfilling:
                return self.aggregate()

    def retain_baseline(self):
        """Keep the previous full readout as a fallback for unindexed sources."""
        from simready_bench.resume import _unrun

        current = self.output / "results.json"
        directory = self.output / ".coordinator-reports"
        paths = sorted(
            directory.glob("baseline-*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        if current.is_file():
            raw = current.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            directory.mkdir(mode=0o700, exist_ok=True)
            retained = directory / ("baseline-" + digest + ".json")
            if not retained.exists():
                from simready_bench.recovery import _save

                _save(retained, json.loads(raw))
            if retained not in paths:
                paths.append(retained)
        for path in paths:
            value = self._check(self.output, path, self.expected)
            key = str(path)
            self.entries[key] = (
                self.output,
                {"directory": str(self.output), "results": key},
            )
            self.fingerprints[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.sources[key] = {
                "snapshot": path.name,
                "root": str(self.output),
                "sha256": self.fingerprints[key],
            }
            for row in value["results"]:
                identity = row["condition_id"]
                if row.get("status") == "completed":
                    digest = _completed_digest(row)
                    if (
                        identity in self.completed_measurements
                        and self.completed_measurements[identity] != digest
                    ):
                        raise ValueError(
                            "A completed baseline changes retained measurements"
                        )
                    self.completed_sources.setdefault(identity, key)
                    self.completed_measurements[identity] = digest
                if not _unrun(row):
                    self.closed.add(identity)
        if paths:
            self._save_index()

    def backfill(self, batches):
        with self.lock:
            self.backfilling = True
            for batch in batches:
                for root, report, _ in report_sources(batch):
                    self.add(batch, {"directory": str(root), **report}, defer=True)
            self.retain_baseline()
            self.backfilling = False

    def aggregate(self, *, final=False):
        from simready_bench.benchmark import _aggregate_reports
        from simready_bench.recovery import _save
        from simready_bench.reporting import read_generated_report

        with self.lock:
            if self.backfilling:
                return self.result
            selected = {}
            for key, (_, report) in self.entries.items():
                for row in read_generated_report(report["results"])["results"]:
                    identity = row["condition_id"]
                    if identity not in self.completed_sources:
                        selected[identity] = key
            selected.update(self.completed_sources)
            used = set(selected.values())
            signature = hashlib.sha256(
                json.dumps(
                    sorted((key, self.fingerprints[key]) for key in used)
                ).encode()
            ).hexdigest()
            if signature == self.published_signature and not final:
                return self.result
            self.result = _aggregate_reports(
                [entry for key, entry in self.entries.items() if key in used],
                output_dir=self.output,
                planned_conditions=len(self.expected),
                condition_sources=selected,
            )
            state_path = self.output / "publication-state.json"
            if self.publisher is not None and used:
                if time.time() < self.retry_after and signature == getattr(
                    self, "failed_signature", None
                ):
                    return self.result
                try:
                    self.publisher(
                        self.result,
                        final=final
                        and len(self.completed_sources) == len(self.expected),
                    )
                except Exception as error:
                    self.publication_pending = True
                    self.failed_signature = signature
                    self.retry_after = time.time() + 5
                    _save(
                        state_path,
                        {
                            "schema": "simready-bench-publication-state-v1",
                            "status": "pending",
                            "error_type": type(error).__name__,
                            "completed_conditions": sorted(self.completed_sources),
                            "retry_after": self.retry_after,
                        },
                    )
                    return self.result
            self.published_signature = signature
            self.publication_pending = False
            self.retry_after = 0
            _save(
                state_path,
                {
                    "schema": "simready-bench-publication-state-v1",
                    "status": "published" if self.publisher is not None else "local",
                    "completed_conditions": sorted(self.completed_sources),
                },
            )
            return self.result


def coordinate_daytona(
    manifest,
    *,
    policy,
    output_dir,
    workers,
    wheel,
    cpu_snapshot,
    runtime_images,
    work_dir=None,
    on_report=None,
    max_retries=2,
    retry_delay_seconds=5,
    maximum_retry_delay_seconds=60,
    dispatch_pending=True,
    allowed_pending=None,
):
    """Replace a finished slot while unrelated saved executors keep running."""
    from simready_bench.daytona import (
        DaytonaObservationError,
        reconnect_daytona,
        run_daytona,
    )
    from simready_bench.parallel import _stage_shards
    from simready_bench.recovery import (
        RecoveryBudgetExhausted,
        RecoveryLimitReached,
        _save,
        validate_recovery,
    )
    from simready_bench.reporting import read_generated_report
    from simready_bench.resume import (
        _batches,
        _daytona_client,
        _read,
        _remaining,
        _retire,
        _reports,
        condition_plan,
    )

    validate_recovery(max_retries, retry_delay_seconds, maximum_retry_delay_seconds)
    output = Path(os.path.abspath(output_dir))
    expected = condition_plan(manifest)
    index = ReportIndex(output, manifest, on_report)
    client = None
    state_path = output / "coordinator-state.json"
    retry_policy = {
        "max_retries": max_retries,
        "retry_delay_seconds": retry_delay_seconds,
        "maximum_retry_delay_seconds": maximum_retry_delay_seconds,
    }
    if state_path.exists():
        state = _read(state_path)
        if (
            state.get("manifest_sha256")
            != hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        ):
            raise ValueError("The saved coordinator matrix cannot change")
        if state.get("retry_policy") != retry_policy:
            raise ValueError("The saved coordinator retry policy cannot change")
    else:
        state = {
            "schema": "simready-bench-coordinator-v1",
            "manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
            "dispatches": {},
            "observation_failures": {},
            "not_before": {},
            "retry_policy": retry_policy,
        }
    batches, finished, futures = {}, set(), {}
    blocked = set()
    errors = []
    with _observer_guard:
        previous = _live_observers.get(str(output))
        if previous is not None and any(
            thread.is_alive() for thread in previous.threads
        ):
            raise DaytonaObservationError(
                "Previous host observers still own this output in this process"
            )
        pool = _Observers()
        _live_observers[str(output)] = pool
    interrupted = False

    def get_client():
        nonlocal client
        if client is None:
            client = _daytona_client()
        return client

    def save(status):
        state.update(
            status=status, active_batches=[batch.name for batch in futures.values()]
        )
        _save(state_path, state)

    def execute(batch):
        selected = batches[batch]
        def callback(report, **_):
            return index.add(batch, report)
        if (batch / "daytona-delivery.json").is_file():
            return reconnect_daytona(
                selected,
                policy=policy,
                output_dir=batch,
                client=get_client(),
                on_report=callback,
            )
        selected_policy = _read(batch / "operator.json")
        return run_daytona(
            selected,
            policy=selected_policy,
            output_dir=batch,
            assets_root=batch / "assets",
            skills_root=batch / "skills",
            work_dir=work_dir,
            wheel=wheel,
            cpu_snapshot=cpu_snapshot,
            runtime_images=runtime_images,
            on_report=callback,
        )

    try:
        while True:
            active = set(futures.values())
            owned = set()
            for batch in _batches(output):
                if batch == output and not (batch / "daytona-delivery.json").exists():
                    continue
                selected = (
                    manifest if batch == output else _read(batch / "experiment.json")
                )
                baseline, actual = copy.deepcopy(manifest), copy.deepcopy(selected)
                baseline.pop("reconstruction_selection", None)
                actual.pop("reconstruction_selection", None)
                if (
                    baseline != actual
                    or not set(condition_plan(selected)) <= expected.keys()
                ):
                    raise ValueError("A saved executor changes the original matrix")
                batches[batch] = selected
                delivery_path = batch / "daytona-delivery.json"
                if (
                    delivery_path.exists()
                    and _read(delivery_path).get("schema")
                    == "simready-bench-daytona-delivery-v2"
                ):
                    state["dispatches"].setdefault(
                        batch.name, list(condition_plan(selected))
                    )
            index.backfill(batches)
            for batch, selected in batches.items():
                if batch in finished:
                    continue
                path = batch / "daytona-delivery.json"
                delivery = _read(path) if path.exists() else None
                if (
                    delivery
                    and delivery.get("cleanup_confirmed")
                    and delivery.get("collection") in ("completed", "not_started")
                ):
                    _retire(
                        batch,
                        _reports(batch),
                        policy=policy,
                        manifest=manifest,
                        client=get_client(),
                    )
                    proof = _read(batch / "retirement.json")
                    index.closed.update(proof.get("admitted_conditions", []))
                    index.closed.update(proof.get("terminal_unadmitted_conditions", []))
                    finished.add(batch)
                    continue
                if (
                    batch not in active
                    and batch.name in state["dispatches"]
                    and delivery is None
                ):
                    finished.add(batch)
                    continue
                identities = set(condition_plan(selected))
                if owned & identities:
                    raise ValueError("Two active executors own the same reconstruction")
                owned.update(identities)
            pending = set(expected) - index.closed - owned - blocked
            if allowed_pending is not None:
                pending &= set(allowed_pending)
            available = workers - len(futures)
            for batch in batches:
                if not available:
                    break
                if batch in finished or batch in active:
                    continue
                if time.time() < state["not_before"].get(batch.name, 0):
                    continue
                if not (batch / "daytona-delivery.json").exists():
                    state["dispatches"].setdefault(
                        batch.name, list(condition_plan(batches[batch]))
                    )
                    save("dispatching")
                futures[pool.submit(execute, batch)] = batch
                available -= 1
            if pending and available and dispatch_pending:
                counts = {
                    identity: sum(
                        identity in entries for entries in state["dispatches"].values()
                    )
                    for identity in pending
                }
                if any(count > max_retries for count in counts.values()):
                    raise RecoveryLimitReached(
                        "Unadmitted work exhausted its executor retries"
                    )
                requests, dollars = _remaining(policy, manifest)
                budget = manifest["defaults"]["budgets"]
                if (
                    len(pending) * budget["inference_requests"] > requests
                    or Decimal(str(budget["maximum_spend_usd"])) * len(pending)
                    > dollars
                ):
                    raise RecoveryBudgetExhausted(
                        "Pending work exceeds the original unspent allowance"
                    )
                selected = copy.deepcopy(manifest)
                selected["reconstruction_selection"] = {
                    asset["id"]: [] for asset in manifest["assets"]
                }
                for identity in pending:
                    row = expected[identity]
                    selected["reconstruction_selection"][row["asset_id"]].append(
                        row["repetition"]
                    )
                for values in selected["reconstruction_selection"].values():
                    values.sort()
                prepared = _stage_shards(
                    selected,
                    policy,
                    output,
                    available,
                    assets_root=policy["roots"]["assets"],
                    skills_root=policy["roots"]["skills"],
                    wheel=wheel,
                    cpu_snapshot=cpu_snapshot,
                    runtime_images=runtime_images,
                )
                for batch, selected, _ in prepared:
                    batches[batch] = selected
                    state["dispatches"][batch.name] = list(condition_plan(selected))
                save("dispatching")
                for batch, selected, _ in prepared:
                    futures[pool.submit(execute, batch)] = batch
            save("running")
            report = index.aggregate()
            if not futures:
                if not pending or not dispatch_pending:
                    break
                time.sleep(min(1, maximum_retry_delay_seconds))
                continue
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                batch = futures.pop(future)
                failure = None
                try:
                    future.result()
                except Exception as error:
                    failure = error
                    errors.append(error)
                    count = state["observation_failures"].get(batch.name, 0) + 1
                    state["observation_failures"][batch.name] = count
                    state["not_before"][batch.name] = time.time() + min(
                        maximum_retry_delay_seconds,
                        retry_delay_seconds * 2 ** min(count - 1, 10),
                    )
                    if isinstance(error, DaytonaObservationError):
                        raise
                if not (batch / "daytona-delivery.json").exists():
                    finished.add(batch)
                    if failure is not None:
                        blocked.update(condition_plan(batches[batch]))
        report = index.aggregate(final=True)
        result = {
            **read_generated_report(report["results"]),
            "report": report,
            "dispatch_pending": dispatch_pending,
        }
        if index.publication_pending:
            error = DaytonaObservationError(
                "Retained results await publication, so resume without repeating authoring"
            )
            error.result = result
            raise error
        save(
            "completed"
            if result["completed_conditions"] == len(expected)
            else "incomplete"
        )
        if result["completed_conditions"] != len(expected) and dispatch_pending:
            error = (
                errors[0]
                if blocked and errors
                else RecoveryLimitReached(
                    "Admitted incomplete attempts retain their original outcomes"
                )
            )
            error.result = result
            raise error
        return result
    except BaseException as error:
        interrupted = True
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            save("interrupted")
            raise
        try:
            report = index.aggregate()
        except Exception:
            report = index.result
        if report is not None:
            error.result = {
                **read_generated_report(report["results"]),
                "report": report,
            }
        save(
            "publication_pending"
            if index.publication_pending
            else (
                "observation_pending"
                if isinstance(error, DaytonaObservationError)
                else "incomplete"
            )
        )
        raise
    finally:
        index.detached = True
        pool.shutdown(wait=not interrupted)
        with _observer_guard:
            if _live_observers.get(str(output)) is pool and not any(
                thread.is_alive() for thread in pool.threads
            ):
                _live_observers.pop(str(output), None)
