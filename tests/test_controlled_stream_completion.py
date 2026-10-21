"""A verified actor stop preserves measured scores without completing its stream."""

import asyncio
from copy import deepcopy

import pytest

from simready_bench.engine.scoring import pinned_profile
from simready_bench.local import complete_trial_result
from test_budget_completion import docker_boundary, evaluate_final, native_result
from test_confirmed_submit_stop import setup


def cancelled_stream(**changes):
    return {
        "status": "cancelled",
        "observed_upstream_status": 200,
        "observed_provider_response_statuses": [],
        "partial_response_bytes": 64233,
        "partial_usage_terminal": False,
        **changes,
    }


@pytest.mark.parametrize("unknown", [False, True])
def test_budget_stop_keeps_measured_zero_but_not_an_unknown_score(
    tmp_path, monkeypatch, unknown
):
    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        for number in range(3):
            controller.reserve_inference("episode", str(number))
        before = controller.get_episode("episode")
        try:
            await bridge.request_stop("episode", "inference_budget_exhausted")
            assert bridge.completed_actor_stop(env)
            assert env.final_snapshot.valid
            evaluation = evaluate_final(
                tmp_path,
                env.final_snapshot,
                unknown=unknown,
                failed=[row["code"] for row in pinned_profile()["mandatory_requirements"]],
            )
            assert evaluation["simready"]["score_percent"] == (
                None if unknown else 0.0
            )
            records = [
                {"status": "completed", "upstream_status": 200},
                {"status": "completed", "upstream_status": 200},
                cancelled_stream(),
            ]
            original_records = deepcopy(records)
            result = complete_trial_result(
                {
                    "evaluation": evaluation,
                    "official_reward": evaluation["official_reward"],
                    "model_quality_evidence": True,
                    "gateway_records": records,
                },
                native_result("NonZeroAgentExitCodeError"),
                normal_stop=bridge.completed_actor_stop(env),
            )
            assert result["status"] == (
                "infrastructure_error" if unknown else "completed"
            )
            assert result["evaluation"] is evaluation
            assert result["official_reward"] == (None if unknown else 0)
            assert result["model_quality_evidence"] is (not unknown)
            assert records == original_records
            assert records[-1]["partial_usage_terminal"] is False
            assert "usage" not in records[-1]
            after = controller.get_episode("episode")
            assert after.inference_calls_used == before.inference_calls_used == 3
            assert after.inference_limit == before.inference_limit
            assert after.deadline_at == before.deadline_at
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "records,normal_stop",
    [
        ([cancelled_stream(), {"status": "completed", "upstream_status": 200}], True),
        ([cancelled_stream(), cancelled_stream()], True),
        ([cancelled_stream()], False),
        ([cancelled_stream(error_code="client_disconnected")], True),
        ([cancelled_stream(error_code="upstream_transport_error")], True),
        ([cancelled_stream(observed_upstream_status=400)], True),
        ([cancelled_stream(observed_upstream_status=429)], True),
        ([cancelled_stream(observed_upstream_status=503)], True),
        ([cancelled_stream(upstream_status=500)], True),
        ([cancelled_stream(provider_response_statuses=["failed"])], True),
        ([cancelled_stream(provider_response_statuses=["cancelled"])], True),
        ([cancelled_stream(observed_provider_response_statuses=["failed"])], True),
        ([cancelled_stream(observed_provider_response_statuses=["cancelled"])], True),
    ],
)
def test_uncertain_streams_do_not_acquire_model_quality_evidence(records, normal_stop):
    records = [{"status": "completed", "upstream_status": 200}, *records]
    result = complete_trial_result(
        {
            "evaluation": {"simready": {"score_percent": 0.0}},
            "official_reward": 0,
            "model_quality_evidence": True,
            "gateway_records": records,
        },
        native_result(None),
        normal_stop=normal_stop,
    )
    assert result["status"] == "infrastructure_error"
    assert result["evaluation"]["simready"]["score_percent"] == 0.0
    assert result["official_reward"] is None
    assert result["model_quality_evidence"] is False


def test_one_interrupted_request_does_not_establish_successful_model_generation():
    result = complete_trial_result(
        {
            "evaluation": {"simready": {"score_percent": 0.0}},
            "gateway_records": [cancelled_stream()],
        },
        native_result(None),
        normal_stop=True,
    )
    assert result["status"] == "infrastructure_error"
    assert result["evaluation"]["simready"]["score_percent"] == 0.0


@pytest.mark.parametrize("upstream_status", [400, 429, 500, 503])
def test_final_stream_stop_does_not_waive_an_earlier_provider_rejection(upstream_status):
    result = complete_trial_result(
        {
            "evaluation": {"simready": {"score_percent": 0.0}},
            "gateway_records": [
                {"status": "completed", "upstream_status": upstream_status},
                {"status": "completed", "upstream_status": 200},
                cancelled_stream(),
            ],
        },
        native_result(None),
        normal_stop=True,
    )
    assert result["status"] == "infrastructure_error"
    assert result["evaluation"]["simready"]["score_percent"] == 0.0
