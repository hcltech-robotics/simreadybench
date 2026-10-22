"""Local matrix workers isolate native hooks and retain one result publisher."""

import asyncio
import json
import os
from pathlib import Path
import threading
import time

import pytest

from simready_bench import local


def _isolated_fixture_condition(condition, prepared, resolved, root, controller_root, throttle):
    markers = root.parent / "worker-pids"
    markers.mkdir(exist_ok=True)
    (markers / str(os.getpid())).touch()
    deadline = time.monotonic() + 10
    while len(list(markers.iterdir())) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    started = time.monotonic()
    time.sleep(0.15)
    if condition.get("fixture_failure"):
        raise ConnectionError("Injected isolated worker failure")
    return {"condition_id": condition["condition_id"], "status": "completed", "worker_pid": os.getpid(), "started": started, "finished": time.monotonic(), "route_key": (throttle or {}).get("route_key")}


def _until_stopped_fixture_condition(condition, prepared, resolved, root, controller_root, throttle):
    if condition["condition_id"] == "condition-0":
        deadline = time.monotonic() + 10
        while not (root.parent / "second-started").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    else:
        (root.parent / "second-started").touch()
        deadline = time.monotonic() + 10
        while not local._worker_stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        (root.parent / "second-cleaned").touch()
    return {"condition_id": condition["condition_id"], "status": "completed"}


@pytest.mark.parametrize("failure", [False, True])
def test_spawn_workers_bound_concurrency_preserve_identity_and_continue_failures(tmp_path, monkeypatch, failure):
    conditions = [{"condition_id": "condition-" + str(index), "asset_id": "asset", "harness_id": "pi", "model_id": "selected", "skill_set_id": "none", "model": {"provider_ref": "selected", "model_id": "fixture-model"}, "settings": {"budgets": {"maximum_spend_usd": 1}}, "fixture_failure": failure and index == 0} for index in range(4)]
    resolved = {"conditions": conditions}
    prepared = {condition["condition_id"]: condition for condition in conditions}
    ledger = local.MatrixLedger(tmp_path / "matrix.sqlite3", "a" * 64, 4)
    monkeypatch.setattr(local, "_condition_process", _isolated_fixture_condition)

    async def execute():
        results = []
        outcomes = local._condition_results(resolved, prepared, tmp_path, object(), ledger, 2, {("selected", "fixture-model"): {"route_key": "route-fixture"}})
        try:
            async for index, condition, result, cached in outcomes:
                assert not cached
                ledger.complete(condition["condition_id"], result)
                results.append(result)
        finally:
            await outcomes.aclose()
        return results

    results = asyncio.run(execute())
    assert {result["condition_id"] for result in results} == set(prepared)
    completed = [result for result in results if result["status"] == "completed"]
    assert len(completed) == (3 if failure else 4)
    assert len({result["worker_pid"] for result in completed}) == 2
    assert all(result["worker_pid"] != os.getpid() for result in completed)
    assert all(result["route_key"] == "route-fixture" for result in completed)
    if not failure:
        events = [(result["started"], 1) for result in completed] + [(result["finished"], -1) for result in completed]
        active = peak = 0
        for _, change in sorted(events):
            active += change
            peak = max(peak, active)
        assert peak == 2
    assert all(record["status"] == "completed" for record in ledger.records())


def test_worker_stop_waits_for_native_condition_cleanup(monkeypatch):
    stop, cleaned = threading.Event(), threading.Event()
    monkeypatch.setattr(local, "_worker_stop", stop)
    monkeypatch.setattr(local, "bounded_controller_class", lambda: lambda root: object())

    async def condition(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.set()

    monkeypatch.setattr(local, "_condition", condition)
    thread = threading.Thread(target=lambda: (time.sleep(0.05), stop.set()))
    thread.start()
    try:
        with pytest.raises(asyncio.CancelledError):
            local._condition_process({}, {}, {}, Path("unused"), Path("unused"), None)
    finally:
        thread.join(2)
    assert cleaned.is_set()


def test_closing_parallel_results_joins_workers_before_returning(tmp_path, monkeypatch):
    conditions = [{"condition_id": "condition-" + str(index), "model": {"provider_ref": "selected", "model_id": "fixture"}, "settings": {"budgets": {"maximum_spend_usd": 1}}} for index in range(4)]
    ledger = local.MatrixLedger(tmp_path / "matrix.sqlite3", "a" * 64, 4)
    monkeypatch.setattr(local, "_condition_process", _until_stopped_fixture_condition)

    async def execute():
        outcomes = local._condition_results({"conditions": conditions}, {condition["condition_id"]: condition for condition in conditions}, tmp_path, object(), ledger, 2, {})
        try:
            first = await anext(outcomes)
            assert first[1]["condition_id"] == "condition-0"
        finally:
            await outcomes.aclose()

    asyncio.run(execute())
    assert (tmp_path / "second-cleaned").is_file()
    assert len(ledger.records()) == 2


def test_storage_guard_precedes_condition_budget_admission(tmp_path, monkeypatch):
    from types import SimpleNamespace

    condition = {"condition_id": "a", "settings": {"budgets": {"maximum_spend_usd": 1}}}
    ledger = local.MatrixLedger(tmp_path / "matrix.sqlite3", "a" * 64, 1)
    monkeypatch.setattr(local.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))

    async def execute():
        outcomes = local._condition_results({"conditions": [condition]}, {}, tmp_path, object(), ledger, 1, {})
        try:
            await anext(outcomes)
        finally:
            await outcomes.aclose()

    with pytest.raises(RuntimeError, match="1 GiB"):
        asyncio.run(execute())
    assert ledger.records() == []


def test_parallel_reporter_uses_one_matrix_and_forwards_workers(tmp_path, monkeypatch):
    from simready_bench import benchmark, controller_runner

    seen = []

    class Reporter:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            pass

    monkeypatch.setattr(controller_runner, "ControllerReporter", Reporter)
    monkeypatch.setattr(benchmark, "execute_local", lambda manifest, **options: seen.append(options) or {"completed_conditions": 100})
    monkeypatch.setattr(benchmark, "execute_local_sequential", lambda *args, **kwargs: pytest.fail("Parallel execution selected the sequential asset path"))
    benchmark.execute_local_reported({"assets": [{"id": "a"}, {"id": "b"}]}, policy={}, output_dir=tmp_path, workers=24)
    assert seen[0]["workers"] == 24
