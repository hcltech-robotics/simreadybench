import hashlib
import json
import os

import pytest

from simready_bench import orchestration, resume
from simready_bench.execution_health import ExecutionHealth
from test_orchestration import fixture_report
from test_resume import portable, write


def retire(batch, *, unreported=()):
    proof = {
        "schema": "simready-bench-executor-terminal-v1", "sandbox_id": "worker",
        "run_id": "run", "observed_at": 200, "controller_exit": 137,
        "reason": "inference_transport_circuit_open", "all_containers_stopped": True,
    }
    path = batch / "executor-terminal-proof.json"
    write(path, proof)
    path.chmod(0o600)
    write(batch / "daytona-delivery.json", {
        "sandbox_id": "worker", "run_id": "run", "controller_exit": 137,
        "termination_reason": proof["reason"], "terminal_confirmed": True,
        "cleanup_confirmed": True,
        "health_policy": {"schema": "simready-bench-executor-health-v1"},
        "executor_terminal_proof_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    write(batch / "retirement.json", {
        "sandbox_id": "worker", "executor_deleted": True,
        "unreported_admitted_conditions": list(unreported),
    })


@pytest.mark.skipif(os.name == "nt", reason="Native terminal receipts require Linux ownership")
@pytest.mark.parametrize("reported", [False, True])
def test_unreported_admitted_failure_counts_with_empty_or_success_reports(tmp_path, reported):
    output, batch, selected, identity, path, _ = fixture_report(tmp_path)
    missing = next(key for key in resume.condition_plan(selected) if key != identity)
    if reported:
        value = json.loads(path.read_text())
        value["results"][0].update(simready_reward=0, termination_reason="inference_budget_exhausted")
        value["results"][0]["simready"]["score_percent"] = 0
        write(path, value)
        original = path.read_bytes()
    else:
        path.unlink()
    retire(batch, unreported=[missing])
    observations = orchestration._health_observations([batch])
    assert [row for row in observations if row["status"] == "infrastructure_error"] == [
        {"attempt_id": "001/@executor", "status": "infrastructure_error"}
    ]
    health = ExecutionHealth(output, now=lambda: 250)
    assert not health.observe(observations, active=True)
    assert health.state["failures_since_completion"] == 1
    restored = ExecutionHealth(output, now=lambda: 300)
    assert not restored.observe(orchestration._health_observations([batch]), active=True)
    assert restored.state["failures_since_completion"] == 1
    if reported:
        assert path.read_bytes() == original
        assert health.state["last_completed_at"] == 250
        assert {"attempt_id": "001/" + identity, "status": "completed"} in observations


def test_reported_failure_has_no_additional_executor_failure(tmp_path):
    output, batch, _, identity, path, _ = fixture_report(tmp_path)
    value = json.loads(path.read_text())
    value["results"][0]["status"] = "infrastructure_error"
    write(path, value)
    health = ExecutionHealth(output, now=lambda: 150)
    assert not health.observe(orchestration._health_observations([batch]), active=True)
    retire(batch)
    observations = orchestration._health_observations([batch])
    assert observations == [{"attempt_id": "001/" + identity, "status": "infrastructure_error"}]
    restored = ExecutionHealth(output, now=lambda: 250)
    assert not restored.observe(observations, active=True)
    assert restored.state["failures_since_completion"] == 1


@pytest.mark.skipif(os.name == "nt", reason="Native terminal receipts require Linux ownership")
def test_earlier_failure_and_success_do_not_hide_later_unreported_failure(tmp_path):
    output, batch, selected, identity, path, _ = fixture_report(tmp_path)
    selected["repetitions"] = 4
    write(batch / "experiment.json", selected)
    others = [(key, value) for key, value in resume.condition_plan(selected).items() if key != identity]
    success, success_condition = others[0]
    missing = others[1][0]
    first = json.loads(path.read_text())
    first["results"][0]["status"] = "infrastructure_error"
    write(path, first)
    health = ExecutionHealth(output, now=lambda: 150)
    assert not health.observe(orchestration._health_observations([batch]), active=True)
    assert health.state["failures_since_completion"] == 1
    portable(batch / "reconstructions" / success, [(success, success_condition, "completed")])
    assert not health.observe(orchestration._health_observations([batch]), active=True)
    assert health.state["failures_since_completion"] == 0
    retire(batch, unreported=[missing])
    observations = orchestration._health_observations([batch])
    assert {"attempt_id": "001/" + identity, "status": "infrastructure_error"} in observations
    assert {"attempt_id": "001/@executor", "status": "infrastructure_error"} in observations
    restored = ExecutionHealth(output, now=lambda: 250)
    assert not restored.observe(observations, active=True)
    assert restored.state["failures_since_completion"] == 1
    assert not restored.observe(observations, active=True)
    assert restored.state["failures_since_completion"] == 1


@pytest.mark.skipif(os.name == "nt", reason="Native terminal receipts require Linux ownership")
def test_five_retired_unreported_executors_open_persistent_circuit(tmp_path):
    output, first, _, identity, path, _ = fixture_report(tmp_path)
    path.unlink()
    batches = [first] + [output / "executions" / f"{number:03d}" for number in range(2, 6)]
    for batch in batches:
        retire(batch, unreported=[identity])
    observations = orchestration._health_observations(batches)
    assert len(observations) == 5
    assert len({row["attempt_id"] for row in observations}) == 5
    health = ExecutionHealth(output, now=lambda: 250)
    assert health.observe(observations, active=True)
    assert health.state["reason"] == "repeated_infrastructure_failures"
    assert health.state["failures_since_completion"] == 5
    restored = ExecutionHealth(output, now=lambda: 300)
    assert restored.observe(orchestration._health_observations(batches), active=True)
    assert restored.state["failures_since_completion"] == 5


def test_sealed_completion_is_not_missing_even_when_retirement_lists_it(tmp_path):
    output, batch, _, identity, path, _ = fixture_report(tmp_path)
    original = path.read_bytes()
    retire(batch, unreported=[identity])
    observations = orchestration._health_observations([batch])
    assert observations == [{"attempt_id": "001/" + identity, "status": "completed"}]
    assert path.read_bytes() == original
    health = ExecutionHealth(output)
    assert not health.observe(observations, active=True)
    assert health.state["failures_since_completion"] == 0


def test_post_completion_stop_without_missing_results_is_not_a_failure(tmp_path):
    output, batch, _, identity, _, _ = fixture_report(tmp_path)
    retire(batch)
    observations = orchestration._health_observations([batch])
    assert observations == [{"attempt_id": "001/" + identity, "status": "completed"}]
    health = ExecutionHealth(output)
    assert not health.observe(observations, active=True)
    assert health.state["failures_since_completion"] == 0


def test_normal_turn_exhaustion_and_measured_zero_are_not_executor_failures(tmp_path):
    output, batch, _, _, path, _ = fixture_report(tmp_path)
    value = json.loads(path.read_text())
    value["results"][0].update(termination_reason="inference_budget_exhausted", simready_reward=0)
    value["results"][0]["simready"]["score_percent"] = 0
    write(path, value)
    write(batch / "daytona-delivery.json", {"cleanup_confirmed": True, "controller_exit": 0})
    write(batch / "retirement.json", {
        "sandbox_id": "worker", "executor_deleted": True, "unreported_admitted_conditions": [],
    })
    observations = orchestration._health_observations([batch])
    assert [row["status"] for row in observations] == ["completed"]
    health = ExecutionHealth(output)
    assert not health.observe(observations, active=True)
    assert health.state["failures_since_completion"] == 0
