import asyncio
import sqlite3
import time

import aiohttp
from aiohttp import web
import pytest

from simready_bench.engine.inference_gateway import GatewayConfig, ProjectBudget
from simready_bench.engine.request_cost import native_output_facts
from test_gateway_terminal_response import (
    serve,
    CompletedHandlerGateway,
    actor_and_controller,
    configuration,
    request,
)


@pytest.mark.parametrize("failure_status", [200, 400, 429, 500, 503])
def test_failed_inference_releases_model_round_but_retains_attempt(tmp_path, monkeypatch, failure_status):
    async def run():
        _, controller = actor_and_controller(tmp_path, 1)
        generated = []
        app = web.Application()

        async def generate(incoming):
            generated.append(await incoming.json())
            if len(generated) == 1:
                return web.json_response({"error": {"message": "Inference failed"}}, status=failure_status)
            return web.json_response({"status": "completed", "output": []})

        async def reserve(self, payload, *args, **kwargs):
            return {"request_sha256": "a" * 64}

        monkeypatch.setattr(CompletedHandlerGateway, "reserve_request_cost", reserve)
        app.router.add_post("/v1/responses", generate)
        async with serve(app) as upstream:
            gateway = CompletedHandlerGateway(controller, configuration(upstream, 1))
            token = gateway.issue_episode_token("episode")
            async with serve(gateway.runner()) as local, aiohttp.ClientSession() as client:
                assert (await request(client, gateway, local, token))[0] == failure_status
                assert controller.get_episode("episode").inference_calls_used == 0
                assert (await request(client, gateway, local, token))[0] == 200
                assert controller.get_episode("episode").inference_calls_used == 1
                assert (await request(client, gateway, local, token))[0] == 429
            assert len(generated) == len(gateway.records()) == 2
            with sqlite3.connect(gateway.db_path) as db:
                assert db.execute("SELECT requests_used,attempts_used FROM budgets").fetchone() == (1, 2)

    asyncio.run(run())


def test_gateway_transport_cap_survives_restart_and_duplicate_settlement(tmp_path):
    from simready_bench.engine.inference_gateway import GatewayDenied

    _, controller = actor_and_controller(tmp_path, 2)
    config = configuration("http://127.0.0.1:1234", 1)
    gateway = CompletedHandlerGateway(controller, config)
    for attempt in range(4):
        record = {"gateway_request_id": str(attempt), "status": "admitted"}
        job, _ = gateway._admit("episode", record)
        record.update(status="completed", upstream_status=429)
        gateway._save(job, record)
        gateway._save(job, record)
        assert controller.get_episode("episode").inference_calls_used == 0
        gateway = CompletedHandlerGateway(controller, config)
    with pytest.raises(GatewayDenied, match="project_attempt_budget_exhausted"):
        gateway._check_request_capacity("episode")
    with sqlite3.connect(gateway.db_path) as db:
        assert db.execute("SELECT requests_used,attempts_used FROM budgets").fetchone() == (0, 4)


def test_controller_failed_attempts_are_bounded_and_inflight_rounds_reserved(tmp_path):
    from simready_bench.engine.controller import AdmissionError

    _, controller = actor_and_controller(tmp_path, 1)
    for attempt in range(4):
        job = controller.reserve_inference("episode", str(attempt))
        assert controller.get_episode("episode").inference_calls_used == 1
        with pytest.raises(AdmissionError, match="inference_budget_exhausted"):
            controller.reserve_inference("episode", "parallel-" + str(attempt))
        controller.complete_job(job.job_id, {"status": "transport_failed"})
        assert controller.get_episode("episode").inference_calls_used == 0
    with pytest.raises(AdmissionError, match="inference_attempt_budget_exhausted"):
        controller.reserve_inference("episode", "one-too-many")


@pytest.mark.parametrize("provider_status", [200, 400])
def test_sol_loopback_counts_reserves_and_settles_only_successful_generation(
    tmp_path, provider_status
):
    async def run():
        calls = {"count": 0, "generate": 0}
        app = web.Application()

        async def counter(request):
            calls["count"] += 1
            payload = await request.json()
            assert payload["model"] == "gpt-6.1-sol"
            return web.json_response(
                {"object": "response.input_tokens", "input_tokens": 2000}
            )

        async def generate(request):
            calls["generate"] += 1
            payload = await request.json()
            assert payload["reasoning"]["effort"] == "low"
            if provider_status != 200:
                return web.json_response(
                    {"error": {"message": "Unsupported reasoning"}},
                    status=provider_status,
                )
            return web.json_response(
                {
                    "model": "gpt-6.1-sol",
                    "output": [],
                    "usage": {
                        "input_tokens": 2000,
                        "output_tokens": 7,
                        "total_tokens": 2007,
                    },
                }
            )

        app.router.add_post("/v1/responses/input_tokens", counter)
        app.router.add_post("/v1/responses", generate)
        _, controller = actor_and_controller(tmp_path, 32)
        async with serve(app) as upstream:
            budget = ProjectBudget(
                "simready-bench",
                "run-budget",
                True,
                upstream,
                "gpt-6.1-sol",
                time.time() + 120,
                32,
                2.5,
                "known",
            )
            cfg = GatewayConfig(
                upstream + "/v1",
                budget.model,
                "fixture",
                budget=budget,
                native_output_limits=True,
                max_output_tokens=None,
                max_input_tokens=native_output_facts(budget.model)[
                    "maximum_input_tokens"
                ],
            )
            gateway = CompletedHandlerGateway(controller, cfg)
            token = gateway.issue_episode_token("episode")
            async with (
                serve(gateway.runner()) as local,
                aiohttp.ClientSession() as client,
            ):
                async with client.post(
                    local + "/v1/responses",
                    json={
                        "model": budget.model,
                        "input": "Reconstruct this asset",
                        "reasoning": {"effort": "low"},
                    },
                    headers={"Authorization": "Bearer " + token},
                ) as response:
                    assert response.status == provider_status, await response.text()
                    await response.read()
                await asyncio.wait_for(gateway.handler_finished.get(), 2)
            assert calls == {"count": 1, "generate": 1}
            (record,) = gateway.records("episode")
            assert record["reserved_upper_cost_micro_usd"] > 0
            assert record["upstream_status"] == provider_status
            with sqlite3.connect(controller.state_dir / "request-cost.sqlite3") as db:
                assert (
                    db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 1
                )
                assert db.execute("SELECT COUNT(*) FROM cost_settlements").fetchone()[
                    0
                ] == (provider_status == 200)
            if provider_status == 200:
                assert (
                    record["project_usage_settlement"]["settled_upper_micro_usd"]
                    == 5070
                )
            else:
                assert "project_usage_settlement" not in record

    asyncio.run(run())
