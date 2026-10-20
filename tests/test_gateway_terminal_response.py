"""Local HTTP regressions for final-response tools and fail-closed inference caps.

The upstream is a deterministic loopback fixture, never a model/provider.
Controller state, monetary holds, HTTP streaming and final snapshots are real.
"""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import aiohttp
from aiohttp import web
import pytest

from simready_bench.engine.controller import Controller
from simready_bench.engine.inference_gateway import (
    GatewayConfig,
    InferenceGateway,
    ProjectBudget,
)

MODEL = "gpt-6-astra"


@asynccontextmanager
async def serve(app):
    runner = (
        web.AppRunner(app, access_log=None) if isinstance(app, web.Application) else app
    )
    await runner.setup()
    site = (
        runner.tcp_site("127.0.0.1", 0)
        if hasattr(runner, "tcp_site")
        else web.TCPSite(runner, "127.0.0.1", 0)
    )
    await site.start()
    origin = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
    try:
        yield origin
    finally:
        await runner.cleanup()


class CompletedHandlerGateway(InferenceGateway):
    """Test-only barrier: EOF alone does not prove gateway finally has finished."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler_finished = asyncio.Queue()

    async def handle(self, request):
        try:
            return await super().handle(request)
        finally:
            self.handler_finished.put_nowait(request.path)


def upstream_fixture(counted, generated):
    app = web.Application()

    async def counter(request):
        counted.append(await request.json())
        return web.json_response(
            {"object": "response.input_tokens", "input_tokens": 64000}
        )

    async def generate(request):
        payload = await request.json()
        generated.append(payload)
        result = {
            "model": MODEL,
            "output": [
                {
                    "type": "function_call",
                    "name": "write_asset",
                    "arguments": json.dumps(
                        {"name": "asset.usd", "contents": "#usda 1.0\n"}
                    ),
                }
            ],
            "usage": {"input_tokens": 64000, "output_tokens": 16},
        }
        if payload.get("stream"):
            return web.Response(
                text="data: " + json.dumps(result) + "\n\ndata: [DONE]\n\n",
                content_type="text/event-stream",
            )
        return web.json_response(result)

    app.router.add_post("/v1/responses/input_tokens", counter)
    app.router.add_post("/v1/responses", generate)
    return app


def configuration(origin, project_limit):
    approval = ProjectBudget(
        "simready-bench",
        "local-regression-authority",
        True,
        origin,
        MODEL,
        time.time() + 120,
        project_limit,
        20.0,
        "known",
        False,
    )
    return GatewayConfig(
        origin + "/v1",
        MODEL,
        "local-fixture-not-provider-credential",
        budget=approval,
        max_concurrency=4,
    )


def actor_and_controller(tmp_path, inference_limit, deadline=60):
    actor = tmp_path / "actor"
    (actor / "output").mkdir(parents=True)
    controller = Controller(tmp_path / "trusted")
    controller.register_episode(
        "episode",
        actor,
        task_id="local-regression",
        trial_id="local-regression",
        inference_limit=inference_limit,
        deadline_seconds=deadline,
    )
    controller.start_episode("episode")
    return actor, controller


async def request(client, gateway, origin, token, *, stream=False):
    async with client.post(
        origin + "/v1/responses",
        json={
            "model": MODEL,
            "input": "Local fixture only",
            "max_output_tokens": 4096,
            "stream": stream,
        },
        headers={"Authorization": "Bearer " + token},
    ) as response:
        status, body = response.status, await response.read()
    await asyncio.wait_for(gateway.handler_finished.get(), 2)
    return status, body


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "project_limit,inference_limit,reason",
    [
        (4, 8, "project_request_budget_exhausted"),
        (8, 4, "inference_budget_exhausted"),
        (4, 4, "project_request_budget_exhausted"),
    ],
)
def test_final_allowed_response_can_finish_tool_before_fifth_is_rejected_without_cost(
    tmp_path, stream, project_limit, inference_limit, reason
):
    """Catches stopping/freezing on successful call four, or costing denied call five."""

    async def run():
        counted, generated, stops = [], [], []
        actor, controller = actor_and_controller(tmp_path, inference_limit)
        snapshots = []

        async def stop(episode, why):
            stops.append((episode, why))
            controller.close_episode(episode, why)
            if not snapshots:
                snapshots.append(controller.finalize_submission(episode, reason=why))

        async with serve(upstream_fixture(counted, generated)) as upstream:
            gateway = CompletedHandlerGateway(
                controller,
                configuration(upstream, project_limit),
                on_stop_requested=stop,
            )
            token = gateway.issue_episode_token("episode")
            async with (
                serve(gateway.runner()) as local,
                aiohttp.ClientSession() as client,
            ):
                for _ in range(4):
                    status, body = await request(
                        client, gateway, local, token, stream=stream
                    )
                    assert status == 200
                assert stops == [], (
                    "Last admitted inference must not stop the actor before its tools"
                )
                assert controller.get_episode("episode").status == "active"
                assert gateway.stop_requests() == ()

                # Consume the response and execute its deterministic local tool.
                raw = (
                    body.decode().split("data: ", 1)[1].split("\n\n", 1)[0]
                    if stream
                    else body
                )
                returned = json.loads(raw)
                action = json.loads(returned["output"][0]["arguments"])
                (actor / "output" / action["name"]).write_text(action["contents"])

                status, denied = await request(
                    client, gateway, local, token, stream=stream
                )
                assert status == 429 and json.loads(denied)["error"]["code"] == reason

            assert len(generated) == 4
            assert len(counted) == 4, (
                "A denied fifth request must not call the input counter"
            )
            with sqlite3.connect(controller.state_dir / "request-cost.sqlite3") as db:
                assert (
                    db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 4
                )
            assert controller.get_episode("episode").inference_calls_used == 4
            assert stops == [("episode", reason)]
            assert len(gateway.records("episode")) == 4
            assert (Path(snapshots[0].path) / "asset.usd").read_text() == "#usda 1.0\n"

    asyncio.run(run())


def test_walltime_still_stops_actor_after_final_response_without_another_inference(
    tmp_path,
):
    """Catches allowing the last response to disable the independent wall-clock stop."""
    from simready_bench.engine.harbor import NativeLifecycle

    async def run():
        counted, generated = [], []
        actor, controller = actor_and_controller(
            tmp_path, inference_limit=1, deadline=60
        )
        lifecycle = NativeLifecycle(
            controller, tmp_path / "native-actors", deadline_seconds=60
        )
        stopped = []

        async def stop_actor_writes():
            stopped.append(True)

        # The engine operation alone is substituted; real lifecycle/controller
        # closing, budget state, response delivery and immutable capture remain.
        env = SimpleNamespace(
            actor_root=actor,
            final_snapshot=None,
            actor_phase=True,
            stop_actor_writes=stop_actor_writes,
        )
        lifecycle._episodes["episode"] = env
        async with serve(upstream_fixture(counted, generated)) as upstream:
            gateway = CompletedHandlerGateway(
                controller,
                configuration(upstream, 1),
                on_stop_requested=lifecycle.request_stop,
            )
            token = gateway.issue_episode_token("episode")
            async with (
                serve(gateway.runner()) as local,
                aiohttp.ClientSession() as client,
            ):
                status, _ = await request(client, gateway, local, token)
                assert status == 200 and not stopped
                (actor / "output" / "asset.usd").write_text("#usda 1.0\n")
                with controller._connect() as db:
                    db.execute(
                        "UPDATE episodes SET deadline_at=? WHERE episode_id='episode'",
                        (time.time() - 0.1,),
                    )
                await asyncio.wait_for(lifecycle._watch("episode"), 3)
        assert stopped == [True]
        assert (
            controller.get_episode("episode").termination_reason == "walltime_exhausted"
        )
        assert (
            Path(env.final_snapshot.path) / "asset.usd"
        ).read_text() == "#usda 1.0\n"
        assert len(counted) == len(generated) == 1

    asyncio.run(run())


@pytest.mark.parametrize("project_limit,inference_limit", [(4, 8), (8, 4)])
def test_concurrent_excess_attempts_never_reach_counter_or_money_hold(
    tmp_path, project_limit, inference_limit
):
    """Catches a pre-cost cap check racing other in-flight admissions."""
    from dataclasses import replace

    async def run():
        counted, generated = [], []
        _, controller = actor_and_controller(tmp_path, inference_limit)
        async with serve(upstream_fixture(counted, generated)) as upstream:
            gateway = CompletedHandlerGateway(
                controller,
                replace(configuration(upstream, project_limit), max_concurrency=16),
            )
            token = gateway.issue_episode_token("episode")
            async with (
                serve(gateway.runner()) as local,
                aiohttp.ClientSession() as client,
            ):
                outcomes = await asyncio.gather(
                    *(request(client, gateway, local, token) for _ in range(8))
                )
            assert [status for status, _ in outcomes].count(200) == 4
            assert [status for status, _ in outcomes].count(429) == 4
            assert len(generated) == len(counted) == 4
            with sqlite3.connect(controller.state_dir / "request-cost.sqlite3") as db:
                assert (
                    db.execute("SELECT count(*) FROM reservations").fetchone()[0] == 4
                )
            assert controller.get_episode("episode").inference_calls_used == 4

    asyncio.run(run())
