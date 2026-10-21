import json
import os
from pathlib import Path
import sqlite3
import threading
from dataclasses import replace

import pytest

from simready_bench import parallel, recovery, resume
from simready_bench.engine.inference_gateway import ProjectBudget
from simready_bench.local import ParentAuthorityLedger
from test_parallel import manifest
from test_resume import frozen_executor_wheel, portable, stopped_run, write


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Linux allowance ownership")


def arguments(output, **extra):
    return dict(
        reference="NV25",
        model="gpt-6.1-sol",
        harness="pi",
        deploy="daytona",
        output_dir=output,
        **extra,
    )


def test_collection_only_preserves_results_without_any_replacement_dispatch(
    tmp_path, monkeypatch
):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch, fourth="infrastructure_error")
    monkeypatch.setattr(
        "simready_bench.daytona.run_daytona",
        lambda *args, **kwargs: pytest.fail("Unexpected model dispatch"),
    )
    result = resume.resume_benchmark(
        **arguments(output, dispatch_pending=False, workers=8)
    )
    assert result["completed_conditions"] == 3
    assert result["planned_conditions"] == 100
    assert result["dispatch_pending"] is False
    assert not (output / "executions").exists()
    assert not (output / "recovery-state.json").exists()


def test_transient_failed_wave_retries_with_backoff_and_unchanged_arguments(
    tmp_path, monkeypatch
):
    output, original, policy, _ = stopped_run(tmp_path, monkeypatch)
    identities = list(resume.condition_plan(original))
    complete = set(identities[:3])
    calls, waits = [], []
    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr(
        recovery,
        "attempted_conditions",
        lambda _: ({identities[3]: len(calls)} if calls else {}, complete),
    )

    def execute(**options):
        calls.append(options)
        if len(calls) == 1:
            raise RuntimeError("Transient worker failure")
        return {"completed_conditions": 100, "planned_conditions": 100}

    monkeypatch.setattr(resume, "_resume_once", execute)
    result = resume.resume_benchmark(**arguments(output, workers=8, wheel="frozen.whl"))
    assert result["completed_conditions"] == 100
    assert len(calls) == 2 and waits == [5]
    assert calls[0] == calls[1]
    assert not complete & calls[0]["_allowed_pending"]
    assert calls[0]["wheel"] == "frozen.whl"
    assert json.loads((output / "operator.json").read_text()) == policy
    state = json.loads((output / "recovery-state.json").read_text())
    assert state["status"] == "completed" and state["retry_waves"] == 1


def test_retry_waves_are_bounded_and_survive_a_host_restart(tmp_path, monkeypatch):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    calls, waits = [], []
    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("time.sleep", sleep)

    def execute(**options):
        calls.append(options)
        raise RuntimeError("Unavailable before model admission")

    monkeypatch.setattr(resume, "_resume_once", execute)
    with pytest.raises(RuntimeError, match="Unavailable"):
        resume.resume_benchmark(**arguments(output))
    assert len(calls) == 3 and waits == [5, 10]
    state = json.loads((output / "recovery-state.json").read_text())
    assert state["status"] == "retry_exhausted" and state["retry_waves"] == 2
    with pytest.raises(recovery.RecoveryLimitReached):
        resume.resume_benchmark(**arguments(output))
    assert len(calls) == 3


def test_fresh_failure_counts_towards_the_same_retry_bound(tmp_path, monkeypatch):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    calls, waits = [], []
    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("time.sleep", sleep)

    def execute(**options):
        calls.append(options)
        raise RuntimeError("Unavailable before model admission")

    monkeypatch.setattr(resume, "_resume_once", execute)
    with pytest.raises(RuntimeError, match="Unavailable"):
        resume.resume_benchmark(**arguments(output, _initial_failure=True))
    assert len(calls) == 2 and waits == [5, 10]
    state = json.loads((output / "recovery-state.json").read_text())
    assert state["status"] == "retry_exhausted" and state["retry_waves"] == 2


def test_each_later_condition_receives_its_own_retries(tmp_path, monkeypatch):
    output, original, operator, _ = stopped_run(tmp_path, monkeypatch)
    identities = list(resume.condition_plan(original))
    selected = identities[:3]
    completed = set(identities[3:])
    attempts, calls, waits = {}, [], []
    clock = [100.0]
    monkeypatch.setattr("time.time", lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr(
        recovery, "attempted_conditions", lambda _: (dict(attempts), set(completed))
    )

    def execute(**options):
        calls.append(set(options["_allowed_pending"]))
        assert not completed & calls[-1]
        for identity in selected:
            if identity in completed:
                continue
            attempts[identity] = attempts.get(identity, 0) + 1
            if attempts[identity] < 3:
                raise RuntimeError("Transient failure of " + identity)
            completed.add(identity)
        return {"completed_conditions": 100, "planned_conditions": 100}

    monkeypatch.setattr(resume, "_resume_once", execute)
    result = resume.resume_benchmark(**arguments(output))
    assert result["completed_conditions"] == 100
    assert len(calls) == 7 and set(attempts.values()) == {3}
    assert waits == [5, 10, 5, 10, 5, 10]
    assert json.loads((output / "operator.json").read_text()) == operator
    state = json.loads((output / "recovery-state.json").read_text())
    assert state["status"] == "completed" and state["retry_waves"] == 6


def test_unknown_executor_state_never_creates_a_retry(tmp_path, monkeypatch):
    from simready_bench.daytona import DaytonaObservationError

    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    calls = []

    def execute(**options):
        calls.append(options)
        raise DaytonaObservationError("Existing executor state is unknown")

    monkeypatch.setattr(resume, "_resume_once", execute)
    monkeypatch.setattr("time.sleep", lambda _: pytest.fail("Unexpected retry"))
    with pytest.raises(DaytonaObservationError):
        resume.resume_benchmark(**arguments(output))
    assert len(calls) == 1
    assert (
        json.loads((output / "recovery-state.json").read_text())["status"]
        == "observation_pending"
    )


def test_budget_exhaustion_is_terminal_and_keeps_original_allowance(
    tmp_path, monkeypatch
):
    output, _, policy, _ = stopped_run(
        tmp_path, monkeypatch, fourth="infrastructure_error"
    )
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(resume, "_remaining", lambda *args: (0, 0))
    monkeypatch.setattr(
        "simready_bench.daytona.run_daytona",
        lambda *args, **kwargs: pytest.fail("Unexpected model dispatch"),
    )
    with pytest.raises(recovery.RecoveryBudgetExhausted) as caught:
        resume.resume_benchmark(**arguments(output))
    assert caught.value.result["completed_conditions"] == 3
    assert json.loads((output / "operator.json").read_text()) == policy
    assert (
        json.loads((output / "recovery-state.json").read_text())["status"]
        == "budget_exhausted"
    )


def test_terminal_empty_bootstrap_is_retired_without_spending_allowance(
    tmp_path, monkeypatch
):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    ledger = output / "daytona-executor/authority/parent-allowances.sqlite3"
    with sqlite3.connect(ledger) as database:
        database.execute("DELETE FROM allocations")
    monkeypatch.setattr(resume, "_reports", lambda _: [])
    monkeypatch.setattr(
        "simready_bench.daytona.run_daytona",
        lambda *args, **kwargs: pytest.fail("Unexpected dispatch"),
    )
    result = resume.resume_benchmark(**arguments(output, dispatch_pending=False))
    assert result["completed_conditions"] == 0
    receipt = json.loads((output / "retirement.json").read_text())
    assert receipt["requests"] == 0 and receipt["maximum_spend_usd"] == "0"


def test_invalid_resume_callbacks_do_not_block_a_later_valid_report(
    tmp_path, monkeypatch
):
    from simready_bench import benchmark as workflow

    output, original, policy, _ = stopped_run(tmp_path, monkeypatch)
    old_identity, old_condition = next(iter(resume.condition_plan(original).items()))
    published = []
    dispatched = []
    monkeypatch.setattr(
        workflow,
        "publication_callback",
        lambda *args, **kwargs: lambda report, **kw: published.append(
            json.loads(Path(report["results"]).read_text())
        ),
    )

    def execute(selected, **options):
        conditions = resume.condition_plan(selected)
        dispatched.append(set(conditions))
        parent = ProjectBudget(**policy["providers"]["selected"]["project_budget"])
        ParentAuthorityLedger(policy["runtime"]["authority_state_dir"]).reserve(
            parent,
            allocation_id="shard-" + options["output_dir"].name,
            requests=32 * len(conditions),
            maximum_spend_usd=2.5 * len(conditions),
            work_root=options["output_dir"],
        )
        path = portable(
            options["output_dir"] / "attempts/000-recovered",
            [
                (identity, condition, "completed")
                for identity, condition in resume.condition_plan(selected).items()
            ],
        )
        report = {"directory": str(path.parent), "results": str(path)}
        with pytest.raises(ValueError, match="directory and results"):
            options["on_report"]({"directory": str(path.parent)})
        raw = json.loads(path.read_text())
        raw["results"][0]["seed"] += 1
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="owned identity"):
            options["on_report"](report)
        raw["results"][0]["seed"] -= 1
        foreign = {**raw["results"][0], **old_condition, "condition_id": old_identity}
        raw["results"].append(foreign)
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="owned identity"):
            options["on_report"](report)
        raw["results"].pop()
        path.write_text(json.dumps(raw))
        options["on_report"](report)

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    result = resume.resume_benchmark(**arguments(output, max_retries=0))
    assert result["completed_conditions"] == 100
    assert published[-1]["completed_conditions"] == 100
    assert len(dispatched) == 97
    assert all(len(conditions) == 1 for conditions in dispatched)
    assert len(set().union(*dispatched)) == 97
    assert resume._remaining(policy, original) == (0, 0)


def make_draft(output):
    attempt = output / "attempts/000-reference"
    draft = attempt / ".report-export"
    draft.mkdir()
    for child in list(attempt.iterdir()):
        if child != draft:
            child.rename(draft / child.name)
    return attempt, draft


def test_terminal_draft_salvage_retains_hashes_and_never_reruns_completed_work(
    tmp_path, monkeypatch
):
    output, _, policy, _ = stopped_run(tmp_path, monkeypatch)
    attempt, draft = make_draft(output)
    original = resume._tree(draft)
    reports = resume._reports(output)
    assert len(reports) == 1 and reports[0][0] != attempt
    recovered = reports[0][0]
    assert recovered.is_relative_to(output / "recovered-reports")
    assert (recovered / "report.html").is_file()
    assert (
        sum(resume._completed(recovered, row) for row in reports[0][2]["results"]) == 3
    )
    assert resume._tree(draft) == original
    assert resume._reports(output)[0][0] == recovered
    resume._retire(
        output,
        reports,
        policy=policy,
        manifest=json.loads((output / "experiment.json").read_text()),
        client=resume._daytona_client(),
    )
    assert json.loads((output / "retirement.json").read_text())["requests"] == 96


def test_changed_draft_candidate_is_not_salvaged(tmp_path, monkeypatch):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    attempt, draft = make_draft(output)
    (draft / "candidate/asset.usd").write_bytes(b"changed candidate")
    with pytest.raises(ValueError, match="frozen artefacts"):
        recovery.salvage_draft(output, attempt)
    assert not (output / "recovered-reports").exists()


def test_live_export_draft_is_not_salvaged(tmp_path, monkeypatch):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    attempt, _ = make_draft(output)
    record = json.loads((output / "daytona-delivery.json").read_text())
    record["cleanup_confirmed"] = False
    (output / "daytona-delivery.json").write_text(json.dumps(record))
    assert recovery.salvage_draft(output, attempt) is None


def test_malformed_failed_report_does_not_poison_other_worker_publication(
    tmp_path, monkeypatch
):
    selected = manifest()
    selected["assets"] = selected["assets"][:2]
    roots = {}
    for name in ("assets", "skills", "input_views", "private_manifests"):
        path = tmp_path / name
        path.mkdir()
        roots[name] = str(path)
    wheel = tmp_path / "frozen.whl"
    frozen_executor_wheel(wheel)
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    from simready_bench import benchmark as workflow

    provider = workflow._new_provider(
        "openai",
        "gpt-6.1-sol",
        api_key_env=None,
        maximum_spend_usd=20,
        conditions=8,
        authoring_turns=32,
    )
    policy = {
        "roots": roots,
        "providers": {"selected": provider},
        "runtime": {"authority_state_dir": str(authority)},
        "limits": {
            "maximum_spend_usd": 20,
            "maximum_condition_spend_usd": 2.5,
            "maximum_matrix_runs": 8,
            "inference_requests": 32,
        },
    }
    barrier = threading.Barrier(2)
    rejected = threading.Event()
    publications = []

    def execute(shard, **options):
        conditions = resume.condition_plan(shard)
        parent = ProjectBudget(**provider["project_budget"])
        batch = options["output_dir"]
        allocation = "shard-" + batch.name
        ParentAuthorityLedger(authority).reserve(
            parent,
            allocation_id=allocation,
            requests=32 * len(conditions),
            maximum_spend_usd=2.5 * len(conditions),
            work_root=options["output_dir"],
        )
        child_root = batch / "daytona-executor/authority"
        child_root.mkdir(parents=True, mode=0o700)
        child = replace(
            parent,
            authorization_id="cloud-" + allocation,
            upstream_origin="http://127.0.0.1:1234",
            maximum_requests=32 * len(conditions),
            maximum_spend_usd=2.5 * len(conditions),
        )
        for identity in conditions:
            work_root = "/dev/shm/fixture/" + batch.name
            ParentAuthorityLedger(child_root).reserve(
                child,
                allocation_id=resume._digest({
                    "work_root": work_root,
                    "configuration": "c" * 64,
                    "condition": identity,
                }),
                requests=32,
                maximum_spend_usd=2.5,
                work_root=work_root,
            )
        path = portable(
            options["output_dir"] / "attempts/reference",
            [
                (identity, condition, "completed")
                for identity, condition in resume.condition_plan(shard).items()
            ],
        )
        report = {"directory": str(path.parent), "results": str(path)}
        write(batch / "daytona-executor/controller-status.json", {"status": "failed"})
        write(batch / "daytona-delivery.json", {
            "collection": "completed", "cleanup_confirmed": True,
            "sandbox_id": "deleted-fixture-" + batch.name,
            "parent_allocation": {"allocation_id": allocation},
            "planned_requests": 32 * len(conditions),
            "maximum_spend_usd": 2.5 * len(conditions),
        })
        if int(options["output_dir"].name) <= 2:
            barrier.wait(timeout=10)
        if options["output_dir"].name == "001":
            options["on_report"](report)
            with pytest.raises(ValueError, match="directory and results"):
                options["on_report"]({"directory": str(options["output_dir"])})
            rejected.set()
            raise RuntimeError("Worker failed after its valid report")
        assert rejected.wait(timeout=10)
        options["on_report"](report)

    def publish(report, **_):
        rows = json.loads(Path(report["results"]).read_text())["results"]
        publications.append(len(rows))

    class Deleted:
        def get(self, identifier):
            from daytona.common.errors import DaytonaNotFoundError

            raise DaytonaNotFoundError("deleted")

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    monkeypatch.setattr(resume, "_daytona_client", Deleted)
    result = parallel.run_parallel_daytona(
        selected,
        policy=policy,
        output_dir=tmp_path,
        workers=2,
        wheel=wheel,
        on_report=publish,
    )
    assert publications[-1] == 8
    assert result["completed_conditions"] == 8
    from simready_bench.attempt_state import AttemptStore

    attempts = AttemptStore(tmp_path, selected).snapshot()["attempts"]
    assert len(attempts) == 8 and all(row["cleanup"] == "deleted" for row in attempts)
