"""An actor disconnect cannot cancel its authoritative stop and final capture."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from aiohttp import web
import pytest
from harbor.environments.docker.docker import DockerEnvironment

from simready_bench.engine.inference_gateway import InferenceGateway
from simready_bench.local import complete_trial_result
from test_budget_completion import evaluate_final, native_result
from test_confirmed_submit_stop import setup
from test_gateway_terminal_response import configuration, serve


def test_spend_stop_survives_the_killed_http_clients_disconnect(tmp_path, monkeypatch):
    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        running = True
        commands = []
        calls = {"counter": 0, "model": 0}
        disconnected = asyncio.Event()
        handler_cancelled = asyncio.Event()
        writer = None

        class DisconnectedGateway(InferenceGateway):
            async def handle(self, request):
                try:
                    return await super().handle(request)
                except asyncio.CancelledError:
                    handler_cancelled.set()
                    raise

        async def compose(args, **kwargs):
            nonlocal running
            commands.append(list(args))
            if args[0] == "ps":
                return SimpleNamespace(return_code=0, stdout="a" * 64, stderr="")
            assert args == ["kill", "-s", "SIGKILL", "main"]
            running = False
            # The killed actor owns this genuine TCP connection. Closing it
            # cancels its aiohttp handler while Docker's collector is pending.
            writer.close()
            await writer.wait_closed()
            disconnected.set()
            await asyncio.sleep(0.1)
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def engine(*args):
            assert args == ("inspect", "--format", "{{json .State}}", "a" * 64)
            return json.dumps(
                {
                    "Running": running,
                    "Paused": False,
                    "Restarting": False,
                    "OOMKilled": False,
                    "ExitCode": 0 if running else 137,
                }
            ).encode()

        monkeypatch.setattr(env, "_run_docker_compose_command", compose)
        monkeypatch.setattr(env, "_engine", engine)
        app = web.Application()

        async def counter(request):
            calls["counter"] += 1
            return web.json_response(
                {"object": "response.input_tokens", "input_tokens": 64000}
            )

        async def model(request):
            calls["model"] += 1
            raise AssertionError("Exhausted spend must never reach generation")

        app.router.add_post("/v1/responses/input_tokens", counter)
        app.router.add_post("/v1/responses", model)
        watcher = asyncio.create_task(bridge._watch("episode"))
        try:
            async with serve(app) as upstream:
                config = configuration(upstream, 32)
                config = replace(
                    config,
                    budget=replace(config.budget, maximum_spend_usd=0.000001),
                )
                gateway = DisconnectedGateway(
                    controller, config, on_stop_requested=bridge.request_stop
                )
                token = gateway.issue_episode_token("episode")
                async with serve(gateway.runner()) as local:
                    port = int(local.rsplit(":", 1)[1])
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    body = json.dumps(
                        {
                            "model": config.model,
                            "input": "Reconstruct the reference",
                            "max_output_tokens": 4096,
                        }
                    ).encode()
                    writer.write(
                        (
                            "POST /v1/responses HTTP/1.1\r\n"
                            "Host: 127.0.0.1\r\n"
                            "Content-Type: application/json\r\n"
                            f"Authorization: Bearer {token}\r\n"
                            f"Content-Length: {len(body)}\r\n\r\n"
                        ).encode()
                        + body
                    )
                    await writer.drain()
                    try:
                        await asyncio.wait_for(disconnected.wait(), 3)
                    except TimeoutError:
                        pytest.fail(repr(await asyncio.wait_for(reader.read(4096), 1)))
                    await asyncio.wait_for(watcher, 3)
                    await asyncio.wait_for(handler_cancelled.wait(), 3)
                    # Joining from the agent exit hook must use the same stop.
                    assert await bridge.request_stop("episode", "agent_completed")

                assert calls == {"counter": 1, "model": 0}
                assert commands.count(["kill", "-s", "SIGKILL", "main"]) == 1
                assert env._actor_stop_audit["was_running"] is True
                assert env._actor_stop_audit["kill_return_code"] == 0
                assert env._actor_stop_audit["state"]["ExitCode"] == 137
                assert bridge.completed_actor_stop(env)
                receipt = await bridge.confirmed_actor_exit(env, 137)
                assert receipt.snapshot_id == env.final_snapshot.snapshot_id
                assert receipt.snapshot_sha256 == env.final_snapshot.sha256
                assert gateway.stop_requests()[0]["callback_status"] == "delivered"
                assert controller.get_episode("episode").inference_calls_used == 0
                evaluation = evaluate_final(tmp_path, env.final_snapshot)
                result = complete_trial_result(
                    {
                        "evaluation": evaluation,
                        # The live failure followed successful authoring calls.
                        "gateway_records": [
                            {"status": "completed", "upstream_status": 200}
                        ],
                    },
                    native_result("NonZeroAgentExitCodeError"),
                    normal_stop=bridge.completed_actor_stop(env),
                )
                assert result["status"] == "completed"
                assert result["evaluation"]["simready"]["score_percent"] == 100
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            bridge.uninstall()

    asyncio.run(exercise())


def test_cancelled_stop_waiter_does_not_cancel_or_duplicate_the_lifecycle_operation(
    tmp_path, monkeypatch
):
    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def stop():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            env.actor_stopped = True

        monkeypatch.setattr(env, "stop_actor_writes", stop)
        first = asyncio.create_task(
            bridge.request_stop("episode", "project_spend_budget_exhausted")
        )
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(bridge.request_stop("episode", "submitted"))
        try:
            release.set()
            assert await second is True
            assert calls == 1
            assert env.final_snapshot is not None
            assert env.final_snapshot.valid
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)
            bridge.uninstall()

    asyncio.run(exercise())


def test_failed_shared_stop_retries_within_allowance_and_still_tears_down_docker(
    tmp_path, monkeypatch
):
    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        stops, teardown = [], []

        async def stop():
            stops.append(True)
            raise RuntimeError("Docker state unavailable")

        async def destroy(self, delete):
            assert self is env
            teardown.append(delete)

        monkeypatch.setattr(env, "stop_actor_writes", stop)
        monkeypatch.setattr(DockerEnvironment, "stop", destroy)
        try:
            with pytest.raises(RuntimeError, match="Docker state unavailable"):
                await bridge.request_stop("episode", "submitted")
            with pytest.raises(RuntimeError, match="Docker state unavailable"):
                await env.stop(delete=True)
            assert stops == [True, True]
            assert teardown == [True]
            assert env.final_snapshot is None
            assert await bridge.confirmed_actor_exit(env, 137) is None
            assert not bridge.completed_actor_stop(env)
        finally:
            bridge.uninstall()

    asyncio.run(exercise())
