"""Infrastructure retries are isolated, finite and retain successful attempts."""

import json
import os
import time

import pytest

from simready_bench import daytona, orchestration, recovery, resume
from simready_bench.runtime_images import DEFAULT_IMAGES
from simready_bench.local import ParentAuthorityLedger
from test_daytona_recovery import setup_run, source_failure
from test_parallel import manifest
from test_resume import portable, write
from test_local_resume_budget import interrupted_local_run as _local_run_fixture

interrupted_local_run = _local_run_fixture


@pytest.mark.parametrize("request_capacity", [None, 128])
def test_failed_condition_retries_do_not_repeat_zero_or_stop_other_conditions(
    tmp_path, monkeypatch, request_capacity
):
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    plan = resume.condition_plan(selected)
    retry_id, fail_id, zero_id, reject_id = plan
    output = tmp_path / "run"
    output.mkdir()
    roots = {}
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (output / name).mkdir()
        roots[name] = str(output / name)
    calls = {identity: [] for identity in plan}
    zero_source = []
    monkeypatch.setattr(resume, "_daytona_client", object)
    monkeypatch.setattr(resume, "_retire", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        resume, "_remaining",
        lambda *_: (
            100000 if request_capacity is None else request_capacity - 32 * sum(map(len, calls.values())),
            100000,
        ),
    )

    def execute(shard, **options):
        batch = options["output_dir"]
        rows = []
        chosen = resume.condition_plan(shard)
        for identity, condition in chosen.items():
            calls[identity].append(batch.name)
            count = len(calls[identity])
            status = (
                "infrastructure_error"
                if identity == fail_id or identity == retry_id and count == 1
                else "condition_error"
                if identity == reject_id
                else "completed"
            )
            rows.append((identity, condition, status))
        path = portable(batch / "attempts/final", rows)
        value = json.loads(path.read_text())
        for row in value["results"]:
            row["trial_id"] = row["condition_id"] + "-" + batch.name
            if row["condition_id"] == zero_id:
                row["simready_reward"] = 0.0
                row["simready"]["score_percent"] = 0.0
        write(path, value)
        if zero_id in chosen:
            zero_source.append((path, path.read_bytes()))
        write(
            batch / "daytona-delivery.json",
            {
                "schema": "simready-bench-daytona-delivery-v2",
                "sandbox_id": 'owned-' + batch.name,
                "cleanup_confirmed": True,
                "collection": "completed",
            },
        )
        write(
            batch / "retirement.json",
            {
                "executor_deleted": True,
                "sandbox_id": 'owned-' + batch.name,
                "requests": len(chosen) * 32,
                "admitted_conditions": list(chosen),
                "unreported_admitted_conditions": [],
            },
        )
        options["on_report"]({"directory": str(path.parent), "results": str(path)})

    monkeypatch.setattr(daytona, "run_daytona", execute)
    options = dict(
        policy={"roots": roots},
        output_dir=output,
        workers=2,
        wheel=None,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
        max_retries=2,
        retry_delay_seconds=0,
        maximum_retry_delay_seconds=0,
    )
    error_type = (
        recovery.RecoveryLimitReached
        if request_capacity is None else recovery.RecoveryBudgetExhausted
    )
    with pytest.raises(error_type) as caught:
        orchestration.coordinate_daytona(selected, **options)
    result = caught.value.result
    assert result["completed_conditions"] == (2 if request_capacity is None else 1)
    assert {identity: len(attempts) for identity, attempts in calls.items()} == {
        retry_id: 2 if request_capacity is None else 1,
        fail_id: 3 if request_capacity is None else 1,
        zero_id: 1, reject_id: 1,
    }
    assert len({name for names in calls.values() for name in names}) >= 2
    statuses = {row["condition_id"]: row["status"] for row in result["results"]}
    assert statuses == {
        retry_id: "completed" if request_capacity is None else "infrastructure_error",
        fail_id: "infrastructure_error",
        zero_id: "completed", reject_id: "condition_error",
    }
    assert result["exhausted_conditions"][fail_id]["failed_batches"] == calls[fail_id]
    assert all(path.read_bytes() == raw for path, raw in zero_source)
    before_restart = {identity: list(attempts) for identity, attempts in calls.items()}
    with pytest.raises(error_type):
        orchestration.coordinate_daytona(selected, **options)
    assert calls == before_restart


@pytest.mark.skipif(os.name == "nt", reason="Native condition ownership requires Linux")
def test_local_condition_exception_and_unscored_result_do_not_abort_remaining_rows(
    interrupted_local_run, monkeypatch
):
    from simready_bench import local

    _, _, policy, _, resolved, plan, work = interrupted_local_run
    identities = list(plan)
    calls = []

    async def condition(entry, *_):
        identity = entry["condition_id"]
        calls.append(identity)
        if identity == identities[0]:
            raise ConnectionError("Injected condition transport failure")
        return {
            **plan[identity],
            "condition_id": identity,
            "trial_id": identity,
            "status": "infrastructure_error" if identity == identities[1] else "completed",
            "evaluation": {"simready": {"score_percent": None}},
            "fidelity": {"status": "measured", "score_percent": 50},
        }

    monkeypatch.setattr(local, "_condition", condition)
    result = local.run(resolved, work_dir=work, policy=policy)
    assert calls == identities
    assert result["attempted_conditions"] == 4
    assert result["completed_conditions"] == 2
    assert [row["status"] for row in result["results"]] == [
        "infrastructure_error", "infrastructure_error", "completed", "completed",
    ]
    assert result["results"][1]["fidelity"]["status"] == "measured"


@pytest.mark.skipif(os.name == "nt", reason="Native allowance ownership requires Linux")
@pytest.mark.parametrize("guard", ["valid", "missing", "wrong_sandbox", "unguarded"])
def test_lost_guarded_executor_charges_its_entire_original_ceiling(
    tmp_path, monkeypatch, guard
):
    output, selected, policy, parent, _ = setup_run(tmp_path, monkeypatch)
    batch = output / "executions/001"
    write(batch / "experiment.json", selected)
    authority = ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])
    authority.reserve(
        parent,
        allocation_id="lost-allocation",
        requests=128,
        maximum_spend_usd=10,
        work_root=batch,
    )
    delivery = {
        "schema": "simready-bench-daytona-delivery-v2",
        "collection": "lost",
        "terminal_confirmed": True,
        "cleanup_confirmed": True,
        "sandbox_id": "lost-sandbox",
        "parent_allocation": {"allocation_id": "lost-allocation"},
        "planned_requests": 128,
        "maximum_spend_usd": 10,
        "health_policy": {
            "schema": "simready-bench-executor-health-v1",
            "stall_seconds": 3900,
            "orphan_stop_minutes": 10,
        },
        "missing_executor_proof": {
            "schema": "simready-bench-daytona-not-found-v1",
            "sandbox_id": "lost-sandbox",
            "observed_at": time.time(),
        },
    }
    if guard == "missing":
        delivery.pop("missing_executor_proof")
    elif guard == "wrong_sandbox":
        delivery["missing_executor_proof"]["sandbox_id"] = "another-sandbox"
    elif guard == "unguarded":
        delivery.pop("health_policy")
    write(batch / "daytona-delivery.json", delivery)
    options = dict(policy=policy, manifest=selected, client=resume._daytona_client())
    if guard != "valid":
        with pytest.raises(ValueError, match="Collect the previous executor"):
            resume._retire(batch, [], **options)
        with authority._db() as db:
            assert db.execute("SELECT * FROM retirements").fetchall() == []
        return
    proof = resume._retire(batch, [], **options)
    assert proof["requests"] == 128
    assert proof["maximum_spend_usd"] == "10"
    assert set(proof["admitted_conditions"]) == set(resume.condition_plan(selected))
    assert proof["admitted_conditions"] == proof["unreported_admitted_conditions"]
    assert proof["usage_status"] == "unavailable_full_original_ceiling_charged"
    assert resume._remaining(policy, selected) == (0, 0)
    assert resume._retire(batch, [], **options) == proof


@pytest.mark.skipif(os.name == "nt", reason="Native allowance ownership requires Linux")
def test_local_resume_does_not_retry_an_independent_condition_rejection(
    interrupted_local_run, monkeypatch
):
    from simready_bench import benchmark as workflow

    output, _, _, _, _, plan, _ = interrupted_local_run
    rejected = next(iter(plan))
    portable(
        output / "attempts/000-initial",
        [
            (identity, condition, "condition_error" if identity == rejected else "completed")
            for identity, condition in plan.items()
        ],
    )

    def forbidden(*_a, **_kw):
        raise AssertionError("A condition rejection must never create another attempt")

    monkeypatch.setattr(workflow, "execute_local_reported", forbidden)
    with pytest.raises(recovery.RecoveryLimitReached) as caught:
        workflow.benchmark(
            "nvidia-25", model="gpt-6.1-sol", deploy="local",
            output_dir=output, repetitions=4, resume=True,
            retry_delay_seconds=0, maximum_retry_delay_seconds=0,
        )
    assert caught.value.result["completed_conditions"] == 3


@pytest.mark.skipif(os.name == "nt", reason="Native allowance ownership requires Linux")
@pytest.mark.parametrize(
    "fault",
    [
        None, "missing_binding", "hash", "permissions", "sandbox", "run",
        "reason", "exit", "running", "epoch", "schema", "unguarded", "missing_file",
    ],
)
def test_stale_collected_controller_needs_bound_private_external_stop_proof(
    tmp_path, monkeypatch, fault
):
    output, selected, policy, parent, _ = setup_run(tmp_path, monkeypatch)
    batch = output / "executions/001"
    write(batch / "experiment.json", selected)
    source_failure(batch, selected, policy, parent)
    write(batch / "daytona-executor/controller-status.json", {"status": "running"})
    delivery_path = batch / "daytona-delivery.json"
    delivery = json.loads(delivery_path.read_text())
    delivery.update(
        run_id="bound-run",
        health_policy={"schema": "simready-bench-executor-health-v1"},
        terminal_confirmed=True,
        controller_exit=137,
        termination_reason="executor_stalled",
    )
    terminal = {
        "schema": "simready-bench-executor-terminal-v1",
        "sandbox_id": delivery["sandbox_id"],
        "run_id": delivery["run_id"],
        "observed_at": time.time(),
        "controller_exit": 137,
        "reason": "executor_stalled",
        "all_containers_stopped": True,
    }
    changes = {
        "sandbox": ("sandbox_id", "another-sandbox"),
        "run": ("run_id", "another-run"),
        "reason": ("reason", "controller_exited"),
        "exit": ("controller_exit", 0),
        "running": ("all_containers_stopped", False),
        "epoch": ("observed_at", float("inf")),
        "schema": ("schema", "untrusted"),
    }
    if fault in changes:
        key, value = changes[fault]
        terminal[key] = value
    proof_path = batch / "executor-terminal-proof.json"
    write(proof_path, terminal)
    proof_path.chmod(0o600)
    import hashlib

    delivery["executor_terminal_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    if fault == "missing_binding":
        delivery.pop("executor_terminal_proof_sha256")
    elif fault == "hash":
        delivery["executor_terminal_proof_sha256"] = "0" * 64
    elif fault == "permissions":
        proof_path.chmod(0o644)
    elif fault == "unguarded":
        delivery.pop("health_policy")
    elif fault == "missing_file":
        proof_path.unlink()
    write(delivery_path, delivery)
    options = dict(policy=policy, manifest=selected, client=resume._daytona_client())
    if fault is not None:
        with pytest.raises(ValueError):
            resume._retire(batch, [], **options)
        return
    proof = resume._retire(batch, [], **options)
    assert proof["executor_terminal_proof"] == terminal
    assert proof["requests"] == 128
    assert proof["maximum_spend_usd"] == "10.0"
    assert set(proof["unreported_admitted_conditions"]) == set(resume.condition_plan(selected))
    assert resume._remaining(policy, selected) == (0, 0)
