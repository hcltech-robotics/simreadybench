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
)


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
