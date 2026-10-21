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


def _health_observations(batches):
    """Include owned executor failures even when shutdown interrupted reporting."""
    from simready_bench.resume import _executor_terminal_proof, _read

    observations = []
    for batch in batches:
        rows = {}
        for _, _, report in report_sources(batch):
            for row in report.get("results", []):
                identity = batch.name + "/" + row["condition_id"]
                rows[identity] = {"attempt_id": identity, "status": row.get("status")}
        delivery_path, retirement_path = batch / "daytona-delivery.json", batch / "retirement.json"
        terminal = None
        if delivery_path.is_file() and retirement_path.is_file():
            delivery, retirement = _read(delivery_path), _read(retirement_path)
            unreported = {batch.name + "/" + identity for identity in
                          retirement.get("unreported_admitted_conditions", [])}
            unreported.difference_update(identity for identity, row in rows.items()
                if row["status"] in ("completed", "infrastructure_error", "condition_error"))
            if (delivery.get("cleanup_confirmed") is True
                    and retirement.get("executor_deleted") is True
                    and retirement.get("sandbox_id") == delivery.get("sandbox_id")
                    and unreported
                    and delivery.get("executor_terminal_proof_sha256")):
                terminal = _executor_terminal_proof(batch, delivery)
        if terminal is not None:
            # A sealed retirement separates reported conditions from work that
            # stopped without a result. Earlier failures remain separate events.
            observations.append({"attempt_id": batch.name + "/@executor",
                "status": "infrastructure_error"})
        observations.extend(rows.values())
    return observations


def _completed_digest(row):
    """Bind measurements and content while allowing owned artefact relocation."""
    value = copy.deepcopy(row)
    value.pop("report_path", None)
    for key in ('execution_provenance', 'protocol_sha256', 'run_spec_sha256', 'provenance_coverage'):
        value.pop(key, None)
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


def _closed_condition(row):
    from simready_bench.resume import _unrun

    return row.get("status") != "infrastructure_error" and not _unrun(row)


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
        self.baseline_retained = False
        self.published_signature = None
        self.retry_after = 0
        self.result = None
        self.detached = False
        self.publication_pending = False
        from simready_bench.attempt_state import AttemptStore
        self.attempts = AttemptStore(self.output, manifest)
        self._load_index()
        state_path = self.output / 'publication-state.json'
        self.publication_attempts = 0
        self.publication_held = False
        if state_path.is_file():
            saved = json.loads(state_path.read_text())
            if saved.get('manifest_sha256') == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest():
                if saved.get('status') in ('published', 'local'):
                    self.published_signature = saved.get('delivery_signature')
                else:
                    self.failed_signature = saved.get('delivery_signature')
                    self.retry_after = saved.get('retry_after', 0)
                    self.publication_attempts = saved.get('attempts', 0)
                    self.publication_held = saved.get('status') == 'held'
                    self.publication_pending = True

    def _retain_measurement(self, row, *, batch, root, snapshot):
        identity = row['condition_id']
        digest = _completed_digest(row)
        previous = [item for item in self.attempts.snapshot()['attempts']
                    if item['condition_id'] == identity and item['measurement'] == 'measured']
        attempt_id = previous[0]['attempt_id'] if previous else batch.name + '/' + identity
        if not previous:
            retirement_path = batch / 'retirement.json'
            retirement = json.loads(retirement_path.read_text()) if retirement_path.is_file() else None
            if retirement and not (retirement.get('executor_deleted') and retirement.get('sandbox_id')):
                retirement = None
            self.attempts.lease(identity, attempt_id, str(self.output), retirement=retirement)
        self.attempts.measure(attempt_id, digest)
        self.attempts.update(attempt_id,
            publication=(previous[0]['publication'] if previous and previous[0]['publication'] in {'published', 'held'}
                         else 'pending' if self.publisher is not None else 'not_requested'),
            evidence={'report_directory': str(root), 'snapshot': str(snapshot)})

    def _load_index(self):
        from simready_bench.resume import _digest, _read

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
                    self._retain_measurement(row, batch=root, root=root, snapshot=snapshot)
                    digest = _completed_digest(row)
                    if (
                        identity in self.completed_measurements
                        and self.completed_measurements[identity] != digest
                    ):
                        raise ValueError("Retained completed measurements disagree")
                    self.completed_sources.setdefault(identity, key)
                    self.completed_measurements[identity] = digest
                if _closed_condition(row):
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
            if row.get('infrastructure_hold'):
                error = ValueError('Infrastructure metadata requires an unchanged run specification')
                error.infrastructure_hold = row['infrastructure_hold']
                raise error
            if row.get("status") == "completed" and not _completed(root, row):
                raise ValueError(
                    "A completed reconstruction requires its frozen artefacts"
                )
        return value

    def add(self, batch, report, *, defer=False):
        from simready_bench.resume import _read, condition_plan

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
                    self._retain_measurement(row, batch=batch, root=root, snapshot=snapshot)
                    self.completed_sources.setdefault(identity, key)
                if _closed_condition(row):
                    self.closed.add(identity)
            self._save_index()
            if not defer and not self.backfilling:
                return self.aggregate()

    def retain_baseline(self):
        """Keep the previous full readout as a fallback for unindexed sources."""
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
                if _closed_condition(row):
                    self.closed.add(identity)
        if paths:
            self._save_index()

    def backfill(self, batches):
        with self.lock:
            self.backfilling = True
            for batch in batches:
                for root, report, _ in report_sources(batch):
                    try:
                        self.add(batch, {"directory": str(root), **report}, defer=True)
                    except (ValueError, TypeError) as error:
                        error._simready_batch = batch
                        raise
            if not self.baseline_retained:
                self.retain_baseline()
                self.baseline_retained = True
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
            self.result = _aggregate_reports(
                [entry for key, entry in self.entries.items() if key in used],
                output_dir=self.output,
                planned_conditions=len(self.expected),
                condition_sources=selected,
            )
            portable = read_generated_report(self.result['results'])
            signature = hashlib.sha256(json.dumps({
                'rows': sorted((_completed_digest(row), (row.get('execution_provenance') or {}).get('sha256'))
                               for row in portable['results']),
                'planned': portable.get('planned_conditions'),
                'run_spec': (portable.get('run_spec') or {}).get('sha256'),
                'operational_provenance': (portable.get('operational_provenance') or {}).get('sha256'),
            }, sort_keys=True).encode()).hexdigest()
            if signature == self.published_signature:
                self.publication_pending = False
                return self.result
            state_path = self.output / "publication-state.json"
            if self.publisher is not None and used:
                if signature != getattr(self, 'failed_signature', None):
                    self.publication_attempts = 0
                    self.publication_held = False
                if self.publication_held:
                    self.publication_pending = True
                    self._hold_publication()
                    return self.result
                if time.time() < self.retry_after and signature == getattr(
                    self, "failed_signature", None
                ):
                    return self.result
                try:
                    self.publication_attempts += 1
                    self.publisher(
                        self.result,
                        final=final
                        and len(self.completed_sources) == len(self.expected),
                    )
                except Exception as error:
                    self.publication_pending = True
                    self.failed_signature = signature
                    self.retry_after = time.time() + 5
                    self.publication_held = self.publication_attempts >= 3 or getattr(error, 'retryable', True) is False
                    if self.publication_held:
                        self._hold_publication()
                    _save(
                        state_path,
                        {
                            "schema": "simready-bench-publication-state-v1",
                            "status": 'held' if self.publication_held else 'pending',
                            "error_type": type(error).__name__,
                            "completed_conditions": sorted(self.completed_sources),
                            "retry_after": self.retry_after,
                            'attempts': self.publication_attempts,
                            'delivery_signature': signature,
                            'manifest_sha256': hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest(),
                        },
                    )
                    return self.result
            self.published_signature = signature
            self.publication_pending = False
            if self.publisher is not None:
                for attempt in self.attempts.snapshot()['attempts']:
                    if attempt['condition_id'] in self.completed_sources and attempt['measurement'] == 'measured':
                        self.attempts.update(attempt['attempt_id'], publication='published')
            self.retry_after = 0
            _save(
                state_path,
                {
                    "schema": "simready-bench-publication-state-v1",
                    "status": "published" if self.publisher is not None else "local",
                    "completed_conditions": sorted(self.completed_sources),
                    'delivery_signature': signature,
                    'manifest_sha256': hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest(),
                    'attempts': self.publication_attempts,
                },
            )
            return self.result

    def _hold_publication(self):
        for attempt in self.attempts.snapshot()['attempts']:
            if (attempt['condition_id'] in self.completed_sources
                    and attempt['measurement'] == 'measured'
                    and attempt['publication'] != 'published'):
                self.attempts.update(attempt['attempt_id'], publication='held')


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
    from simready_bench.attempt_state import AttemptStore
    attempts_store = AttemptStore(output, manifest)
    from simready_bench.execution_health import ExecutionHealth
    health = ExecutionHealth(output)
    health.require_dispatch(allowed_pending, workers)
    expected = condition_plan(manifest)
    index = None
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
    transport_failures = {}
    admitted_attempts = {}
    state.setdefault("condition_retry_counts", {})
    state.setdefault("condition_not_before", {})
    state["blocked_conditions"] = {}
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
    stop_event = threading.Event()
    if health.opened:
        stop_event.set()

    def get_client():
        nonlocal client
        if client is None:
            client = _daytona_client()
        return client

    def cleanup_held_batch(batch, selected, identities, reason):
        pending = [attempts_store.get(batch.name + '/' + identity) for identity in identities]
        pending = [row for row in pending if row['cleanup'] != 'deleted']
        if not pending or any(time.time() < row['evidence'].get('cleanup_retry_after', 0) for row in pending):
            return
        permitted = True
        for row in pending:
            deadline = row['evidence'].get('cleanup_deadline_at', time.time() + 600)
            attempts_store.update(row['attempt_id'], evidence={
                'cleanup_deadline_at': deadline,
                'cleanup_retry_after': time.time() + retry_delay_seconds})
            allowed = attempts_store.retry(row['attempt_id'], 'cleanup',
                maximum_retries=3, deadline_at=deadline)
            permitted = allowed and permitted
        if not permitted:
            return
        from simready_bench.daytona import contain_executor
        try:
            proof = contain_executor(selected, policy=policy, output_dir=batch,
                client=get_client(), reason=reason)
        except Exception as cleanup_error:
            proof_path = batch / 'collection-hold.json'
            proof = _read(proof_path) if proof_path.is_file() else {}
            for row in pending:
                attempts_store.update(row['attempt_id'],
                    evidence={'cleanup_error_type': type(cleanup_error).__name__})
        if proof.get('executor_deleted') is True and proof.get('sandbox_id'):
            for row in pending:
                attempts_store.release(row['attempt_id'], proof)

    def hold_batch(batch, error, reason):
        stop_event.set()
        identities = list(condition_plan(batches[batch]))
        for identity in identities:
            attempt_id = batch.name + '/' + identity
            attempts_store.lease(identity, attempt_id, str(output))
            attempts_store.update(attempt_id, phase='held', infrastructure='held',
                evidence={'failure_phase': 'collection', 'error_type': type(error).__name__,
                    'hold_reason': reason, 'infrastructure_hold': getattr(error, 'infrastructure_hold', None)})
        cleanup_held_batch(batch, batches[batch], identities, reason)
        save('collection_held')

    def save(status):
        state.update(
            status=status,
            active_batches=[batch.name for batch in futures.values()],
            condition_attempts={
                identity: sorted(attempts)
                for identity, attempts in admitted_attempts.items()
            },
        )
        _save(state_path, state)

    def execute(batch):
        selected = batches[batch]
        for identity in condition_plan(selected):
            attempt_id = batch.name + '/' + identity
            attempts_store.lease(identity, attempt_id, str(output))
            attempts_store.update(attempt_id, phase='preparing')

        def callback(report, **_):
            return index.add(batch, report)

        if (batch / "daytona-delivery.json").is_file():
            return reconnect_daytona(
                selected,
                policy=policy,
                output_dir=batch,
                client=get_client(),
                on_report=callback,
                stop_requested=stop_event.is_set,
            )
        if set(condition_plan(selected)) & attempts_store.snapshot()['measurements'].keys():
            raise ValueError('A measured reconstruction cannot create another executor without its saved delivery')
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
            stop_requested=stop_event.is_set,
        )

    try:
        held = [row for row in attempts_store.snapshot()['attempts'] if row['infrastructure'] == 'held']
        if held:
            for batch in _batches(output):
                rows = [row for row in held if row['attempt_id'].startswith(batch.name + '/')]
                if rows:
                    selected = manifest if batch == output else _read(batch / 'experiment.json')
                    cleanup_held_batch(batch, selected, [row['condition_id'] for row in rows],
                        rows[0]['evidence'].get('hold_reason', 'collection_integrity_failure'))
            raise RecoveryLimitReached('Collection or observation is held with retained evidence')
        index = ReportIndex(output, manifest, on_report)
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
                observed_delivery = _read(batch / 'daytona-delivery.json') if (batch / 'daytona-delivery.json').is_file() else {}
                if (batch not in finished and observed_delivery.get('cleanup_confirmed')
                        and observed_delivery.get('collection') in ('completed', 'lost', 'not_started')):
                    _retire(batch, _reports(batch), policy=policy, manifest=manifest, client=get_client())
                for identity in condition_plan(selected):
                    attempt_id = batch.name + '/' + identity
                    retirement_path = batch / 'retirement.json'
                    retired = _read(retirement_path) if retirement_path.is_file() else None
                    if retired and not (retired.get('executor_deleted') is True and retired.get('sandbox_id')):
                        retired = None
                    if observed_delivery.get('cleanup_confirmed') and retired is None:
                        continue
                    attempts_store.lease(identity, attempt_id, str(output), retirement=retired)
                    delivery = batch / 'daytona-delivery.json'
                    if delivery.is_file():
                        observed = _read(delivery)
                        if observed.get('sandbox_id'):
                            attempts_store.update(attempt_id, evidence={'sandbox_id': observed['sandbox_id']})
                delivery_path = batch / "daytona-delivery.json"
                if (
                    delivery_path.exists()
                    and _read(delivery_path).get("schema")
                    == "simready-bench-daytona-delivery-v2"
                ):
                    state["dispatches"].setdefault(
                        batch.name, list(condition_plan(selected))
                    )
            try:
                index.backfill(batches)
            except (ValueError, TypeError) as error:
                failed_batch = getattr(error, '_simready_batch', None)
                for candidate in ([failed_batch] if failed_batch in batches else list(batches)):
                    hold_batch(candidate, error, 'collection_integrity_failure')
                raise RecoveryLimitReached('Collection integrity failed with retained evidence') from error
            observations = _health_observations(batches)
            if health.observe(observations, active=bool(futures)):
                stop_event.set()
                state['health_circuit'] = dict(health.state)
            for batch, selected in batches.items():
                if batch in finished:
                    continue
                path = batch / "daytona-delivery.json"
                delivery = _read(path) if path.exists() else None
                if (
                    delivery
                    and delivery.get("cleanup_confirmed")
                    and delivery.get("collection") in ("completed", "not_started", "lost")
                ):
                    _retire(
                        batch,
                        _reports(batch),
                        policy=policy,
                        manifest=manifest,
                        client=get_client(),
                    )
                    proof = _read(batch / "retirement.json")
                    if proof.get('executor_deleted') is True and proof.get('sandbox_id'):
                        for identity in condition_plan(selected):
                            attempts_store.release(batch.name + '/' + identity, proof)
                    index.closed.update(proof.get("terminal_unadmitted_conditions", []))
                    for identity in proof.get("admitted_conditions", []):
                        attempts = admitted_attempts.setdefault(identity, set())
                        attempts.add(batch.name)
                        count = len(attempts)
                        if count > state["condition_retry_counts"].get(identity, 0):
                            state["condition_retry_counts"][identity] = count
                            state["condition_not_before"][identity] = time.time() + min(
                                maximum_retry_delay_seconds,
                                retry_delay_seconds * 2 ** min(count - 1, 10),
                            )
                    if (
                        proof.get("requests") == 0
                        and not proof.get("admitted_conditions")
                        and not proof.get("unreported_admitted_conditions")
                        and not proof.get("terminal_unadmitted_conditions")
                    ):
                        for identity in condition_plan(selected):
                            transport_failures.setdefault(identity, set()).add(
                                batch.name
                            )
                    finished.add(batch)
                    continue
                identities = set(condition_plan(selected))
                if owned & identities:
                    raise ValueError("Two active executors own the same reconstruction")
                owned.update(identities)
            pending = set(expected) - index.closed - owned - blocked - attempts_store.snapshot()['measurements'].keys()
            if allowed_pending is not None:
                pending &= set(allowed_pending)
            exhausted_admitted = {
                identity
                for identity in pending
                if len(admitted_attempts.get(identity, ())) > max_retries
            }
            for identity in exhausted_admitted:
                state["blocked_conditions"][identity] = {
                    "reason": "condition_infrastructure_retries_exhausted",
                    "failed_batches": sorted(admitted_attempts[identity]),
                    "maximum_retries": max_retries,
                }
            blocked.update(exhausted_admitted)
            pending -= exhausted_admitted
            exhausted = {
                identity
                for identity in pending
                if len(transport_failures.get(identity, ())) > max_retries
            }
            for identity in exhausted:
                state["blocked_conditions"][identity] = {
                    "reason": "unadmitted_transport_retries_exhausted",
                    "failed_batches": sorted(transport_failures[identity]),
                    "maximum_retries": max_retries,
                }
            blocked.update(exhausted)
            pending -= exhausted
            available = workers - len(futures)
            for batch in batches:
                if not available:
                    break
                if batch in finished or batch in active:
                    continue
                if stop_event.is_set() and not (batch / "daytona-delivery.json").exists():
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
            ready = {
                identity
                for identity in pending
                if time.time() >= state["condition_not_before"].get(identity, 0)
            }
            if ready and available and dispatch_pending and not stop_event.is_set() and not index.publication_held:
                requests, dollars = _remaining(policy, manifest)
                budget = manifest["defaults"]["budgets"]
                capacity = requests // budget["inference_requests"]
                condition_dollars = Decimal(str(budget["maximum_spend_usd"]))
                if condition_dollars:
                    capacity = min(
                        capacity, int(Decimal(str(dollars)) // condition_dollars)
                    )
                if capacity <= 0:
                    for identity in ready:
                        state["blocked_conditions"][identity] = {
                            "reason": "project_allowance_exhausted",
                            "failed_batches": sorted(admitted_attempts.get(identity, ())),
                            "maximum_retries": max_retries,
                        }
                    blocked.update(ready)
                    continue
                ready = set(
                    sorted(
                        ready,
                        key=lambda identity: (
                            bool(admitted_attempts.get(identity)),
                            len(admitted_attempts.get(identity, ())),
                            identity,
                        ),
                    )[:min(capacity, available)]
                )
                selected = copy.deepcopy(manifest)
                selected["reconstruction_selection"] = {
                    asset["id"]: [] for asset in manifest["assets"]
                }
                for identity in ready:
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
                if not pending or not dispatch_pending or stop_event.is_set() or index.publication_held:
                    break
                time.sleep(min(1, maximum_retry_delay_seconds))
                continue
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                batch = futures.pop(future)
                completed = False
                try:
                    future.result()
                    completed = True
                except Exception as error:
                    errors.append(error)
                    count = state["observation_failures"].get(batch.name, 0) + 1
                    state["observation_failures"][batch.name] = count
                    state["not_before"][batch.name] = time.time() + min(
                        maximum_retry_delay_seconds,
                        retry_delay_seconds * 2 ** min(count - 1, 10),
                    )
                    identities = list(condition_plan(batches[batch]))
                    deadline = state.setdefault('observation_deadlines', {}).setdefault(
                        batch.name, time.time() + 600)
                    integrity = isinstance(error, (ValueError, TypeError))
                    retry = not integrity
                    retry_phase = 'observation' if isinstance(error, DaytonaObservationError) else 'collection'
                    for identity in identities:
                        attempt_id = batch.name + '/' + identity
                        attempts_store.lease(identity, attempt_id, str(output))
                        retry = attempts_store.retry(attempt_id, retry_phase, maximum_retries=max_retries,
                            deadline_at=deadline) and retry
                    if not retry:
                        hold_batch(batch, error, 'collection_integrity_failure' if integrity else 'collection_retries_exhausted')
                        raise RecoveryLimitReached('Collection exhausted its bounded recovery policy') from error
                    if isinstance(error, DaytonaObservationError):
                        raise
                if completed and not (batch / "daytona-delivery.json").exists():
                    finished.add(batch)
                    blocked.update(set(condition_plan(batches[batch])) - index.closed)
        report = index.aggregate(final=True)
        result = {
            **read_generated_report(report["results"]),
            "report": report,
            "dispatch_pending": dispatch_pending,
            "condition_attempts": {
                identity: sorted(attempts)
                for identity, attempts in admitted_attempts.items()
            },
            "exhausted_conditions": copy.deepcopy(state["blocked_conditions"]),
        }
        if health.opened:
            save("infrastructure_circuit_open")
            error = RecoveryLimitReached("Execution health circuit open: " + health.state['reason'])
            error.result = result
            raise error
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
            if any(
                item["reason"] == "project_allowance_exhausted"
                for item in state["blocked_conditions"].values()
            ):
                error = RecoveryBudgetExhausted(
                    "The original project allowance is exhausted"
                )
            elif state["blocked_conditions"]:
                error = RecoveryLimitReached(
                    "Condition infrastructure retries exhausted"
                    if any(
                        item["reason"] == "condition_infrastructure_retries_exhausted"
                        for item in state["blocked_conditions"].values()
                    )
                    else "Unadmitted work exhausted its transport retries"
                )
            elif blocked and errors:
                error = errors[0]
            else:
                error = RecoveryLimitReached(
                    "Admitted incomplete attempts retain their original outcomes"
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
            report = index.aggregate() if index is not None else None
        except Exception:
            report = index.result if index is not None else None
        if report is not None:
            error.result = {
                **(getattr(error, "result", None) or {}),
                **read_generated_report(report["results"]),
                "report": report,
                "condition_attempts": {
                    identity: sorted(attempts)
                    for identity, attempts in admitted_attempts.items()
                },
                "exhausted_conditions": copy.deepcopy(state["blocked_conditions"]),
            }
        save(
            "publication_pending"
            if index is not None and index.publication_pending
            else (
                "observation_pending"
                if isinstance(error, DaytonaObservationError)
                else "incomplete"
            )
        )
        raise
    finally:
        if index is not None:
            index.detached = True
        pool.shutdown(wait=not interrupted)
        with _observer_guard:
            if _live_observers.get(str(output)) is pool and not any(
                thread.is_alive() for thread in pool.threads
            ):
                _live_observers.pop(str(output), None)
