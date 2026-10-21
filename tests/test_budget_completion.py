"""Configured actor stops preserve final measured reconstruction scores."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from harbor.agents.installed.pi import Pi

from simready_bench.engine.controller import AdmissionError
from simready_bench.engine.evaluation import Evaluator
from simready_bench.engine.harbor import COMPLETION_REASONS
from simready_bench.engine.verifier import scoring_policy
from simready_bench.local import complete_trial_result
from test_confirmed_submit_stop import setup
from test_validator_reward_v1 import official_report


def docker_boundary(env, monkeypatch):
    states = iter(
        [
            {"Running": True, "Paused": False, "OOMKilled": False, "ExitCode": 0},
            {"Running": False, "Paused": False, "OOMKilled": False, "ExitCode": 137},
        ]
    )

    async def compose(args, **kwargs):
        return SimpleNamespace(
            return_code=0, stdout="a" * 64 if args[0] == "ps" else ""
        )

    async def engine(*args):
        return json.dumps(next(states)).encode()

    monkeypatch.setattr(env, "_run_docker_compose_command", compose)
    monkeypatch.setattr(env, "_engine", engine)


def evaluate_final(tmp_path, snapshot, *, unknown=False, failed=()):
    def observe(current, **kwargs):
        report = official_report(failed=failed)
        if unknown:
            report["engine_dispatch_completed"] = False
        return {
            "snapshot_sha256": current.sha256,
            "observation": {
                "source_record": {
                    "sha256": next(
                        row.sha256
                        for row in current.files
                        if row.relative_path == current.entrypoint
                    )
                },
                "nvidia_report": report,
                "preflight": {"package_ok": True},
            },
        }

    evaluator = Evaluator(tmp_path / "evaluator", observer=observe)
    return evaluator.evaluate(
        snapshot,
        {"validator_scoring": scoring_policy()},
        snapshot.episode_id,
        time.time() + 5,
    )


def native_result(error):
    return SimpleNamespace(
        trial_results=[
            SimpleNamespace(
                exception_info=SimpleNamespace(exception_type=error) if error else None,
            )
        ]
    )


@pytest.mark.parametrize("reason", sorted(COMPLETION_REASONS))
def test_configured_stop_finishes_pi_and_keeps_final_score(
    tmp_path, monkeypatch, reason
):
    async def exercise():
        controller, bridge, env, agent = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        before = controller.get_episode("episode")

        async def native_run(self, instruction, environment, context):
            await environment.exec("pi --print --mode json")

        monkeypatch.setattr(Pi, "run", native_run)
        try:
            assert await bridge.request_stop("episode", reason)
            await agent.run("author the reconstruction", env, SimpleNamespace())
            normal_stop = bridge.completed_actor_stop(env)
            assert normal_stop
            assert agent.actor_stop_receipt["termination_reason"] == reason
            evaluation = evaluate_final(tmp_path, env.final_snapshot)
            assert evaluation["simready"]["score_percent"] == 100
            result = {
                "evaluation": evaluation,
                "official_reward": evaluation["official_reward"],
                "model_quality_evidence": True,
                "gateway_records": [
                    {"status": "completed", "upstream_status": 200},
                    {
                        "status": "cancelled",
                        "error_code": (
                            reason
                            if reason
                            in {
                                "project_request_budget_exhausted",
                                "project_spend_budget_exhausted",
                                "inference_budget_exhausted",
                            }
                            else "inference_deadline_exceeded"
                            if reason == "walltime_exhausted"
                            else "episode_admission_closed"
                        ),
                    },
                ],
            }
            complete_trial_result(
                result,
                native_result("NonZeroAgentExitCodeError"),
                normal_stop=normal_stop,
            )
            assert result["status"] == "completed"
            assert result["evaluation"] is evaluation
            assert result["official_reward"] == 1
            assert result["model_quality_evidence"] is True
            after = controller.get_episode("episode")
            assert after.inference_calls_used == before.inference_calls_used
            assert after.inference_limit == before.inference_limit
            assert after.deadline_at == before.deadline_at
            with pytest.raises(AdmissionError):
                controller.reserve_inference("episode", "late-request")
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "failure",
    [
        {"status": "completed", "upstream_status": 400},
        {"status": "completed", "upstream_status": 429},
        {"status": "transport_failed", "error_code": "upstream_transport_error"},
        {
            "status": "completed",
            "upstream_status": 200,
            "provider_response_statuses": ["failed"],
        },
    ],
)
def test_provider_failure_stays_unscored_despite_confirmed_stop(
    tmp_path, monkeypatch, failure
):
    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "project_spend_budget_exhausted")
            evaluation = evaluate_final(tmp_path, env.final_snapshot)
            result = {
                "evaluation": evaluation,
                "official_reward": 1,
                "model_quality_evidence": True,
                "gateway_records": [
                    {"status": "completed", "upstream_status": 200},
                    failure,
                ],
            }
            complete_trial_result(
                result,
                native_result("NonZeroAgentExitCodeError"),
                normal_stop=bridge.completed_actor_stop(env),
            )
            assert result["status"] == "infrastructure_error"
            assert result["evaluation"] is evaluation
            assert result["official_reward"] is None
            assert result["model_quality_evidence"] is False
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "error,unknown,normal_stop",
    [
        ("RuntimeError", False, True),
        ("ApiRateLimitError", False, True),
        ("NonZeroAgentExitCodeError", False, False),
        (None, True, True),
    ],
)
def test_unconfirmed_exit_and_unknown_evaluator_stay_unscored(
    tmp_path, monkeypatch, error, unknown, normal_stop
):
    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "inference_budget_exhausted")
            result = {
                "evaluation": evaluate_final(
                    tmp_path, env.final_snapshot, unknown=unknown
                ),
                "official_reward": 1,
                "gateway_records": [{"status": "completed", "upstream_status": 200}],
            }
            complete_trial_result(result, native_result(error), normal_stop=normal_stop)
            assert result["status"] == "infrastructure_error"
            assert result["evaluation"] is not None
            assert result["official_reward"] is None
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize("limit", ["requests", "time"])
def test_agent_end_closes_at_actual_limit_without_a_thirty_third_request(
    tmp_path, monkeypatch, limit
):
    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        with controller._connect() as db:
            db.execute(
                "UPDATE episodes SET inference_limit=32 WHERE episode_id='episode'"
            )
        if limit == "requests":
            for request in range(32):
                controller.reserve_inference("episode", str(request))
        else:
            with controller._connect() as db:
                db.execute(
                    "UPDATE episodes SET deadline_at=? WHERE episode_id='episode'",
                    (time.time() - 1,),
                )
        try:
            await bridge.on_agent_end(SimpleNamespace(trial_id="episode"))
            episode = controller.get_episode("episode")
            assert episode.termination_reason == (
                "inference_budget_exhausted"
                if limit == "requests"
                else "walltime_exhausted"
            )
            assert episode.inference_calls_used == (32 if limit == "requests" else 0)
            assert episode.inference_limit == 32
            assert bridge.completed_actor_stop(env)
        finally:
            bridge.uninstall()

    asyncio.run(exercise())
