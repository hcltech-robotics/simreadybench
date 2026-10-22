"""Keep route admission time separate from bounded provider response time."""

import asyncio
from dataclasses import replace
import time

import aiohttp
from aiohttp import web
import pytest

from simready_bench.engine import request_cost
from simready_bench.reporting import _statistics, compact_report
from test_provider_overload import relay, Response, SUCCESS
from test_gateway_terminal_response import (
    CompletedHandlerGateway, actor_and_controller, configuration, serve,
)


class Lease:
    lease_seconds = 1200
    def ensure_active(self):
        pass
    def set_on_lost(self, callback):
        pass
    def finish(self, **kwargs):
        pass


def test_queue_longer_than_provider_timeout_still_gets_full_provider_budget(tmp_path):
    with relay(tmp_path, [Response(SUCCESS)], provider_timeout_seconds=100) as (
        state, clock, upstream, post,
    ):
        class Throttle:
            def acquire(self, *, timeout):
                assert timeout == 490
                clock.value += 350
                return Lease()
        state.shared_route_throttle = Throttle()
        status, headers, body = post(deadline=1490)
        assert (status, body) == (200, SUCCESS)
        assert upstream.requests[0][1] == 100
        assert float(headers["X-SimReady-Queue-Wait-Seconds"]) == 350
        assert state.events[0]["queue_wait_seconds"] == 350
        assert state.events[0]["provider_request_seconds"] == 0
        assert state.events[0]["end_to_end_elapsed_seconds"] == 350


def test_queue_wait_preserves_absolute_actor_or_grant_deadline(tmp_path):
    with relay(tmp_path, [Response(SUCCESS)], provider_timeout_seconds=100) as (
        state, clock, upstream, post,
    ):
        class Throttle:
            def acquire(self, *, timeout):
                clock.value += 350
                return Lease()
        state.shared_route_throttle = Throttle()
        assert post(deadline=1400)[0] == 200
        assert upstream.requests[0][1] == 50


def test_exhausted_queue_never_admits_or_calls_provider(tmp_path):
    with relay(tmp_path, [Response(SUCCESS)]) as (state, clock, upstream, post):
        class Throttle:
            def acquire(self, *, timeout):
                clock.value += timeout
                raise TimeoutError("Fixture queue deadline")
        state.shared_route_throttle = Throttle()
        status, headers, _ = post(deadline=1400)
        assert status == 502
        assert float(headers["X-SimReady-Queue-Wait-Seconds"]) == 400
        assert "X-SimReady-Provider-Elapsed-Seconds" not in headers
        assert upstream.requests == []
        assert state.requests == state.calls == 0


def test_gateway_does_not_spend_provider_timeout_waiting_for_relay_permit(tmp_path, monkeypatch):
    async def counter(config, payload, **bounds):
        return request_cost.InputTokenEvidence(
            config.model, 10, request_cost.sha(payload),
            request_cost.sha({key: value for key, value in payload.items()
                              if key in request_cost.COUNT_KEYS}),
            config.upstream_origin, time.time(),
        )
    monkeypatch.setattr(request_cost, "count_input_tokens", counter)

    async def run():
        observed = []
        async def upstream(request):
            observed.append(dict(request.headers))
            await asyncio.sleep(0.25)
            return web.json_response({"model": "gpt-6-astra", "output": []}, headers={
                "X-SimReady-Queue-Wait-Seconds": "0.2",
                "X-SimReady-Provider-Elapsed-Seconds": "0.05",
            })
        app = web.Application()
        app.router.add_post("/v1/responses", upstream)
        async with serve(app) as origin:
            _, controller = actor_and_controller(tmp_path, 2)
            cfg = replace(configuration(origin, 2), request_timeout_seconds=0.1,
                          queue_wait_timeout_seconds=1)
            gateway = CompletedHandlerGateway(controller, cfg)
            gateway._provider_relay_marker = "fixture-relay-marker"
            token = gateway.issue_episode_token("episode")
            async with serve(gateway.runner()) as local, aiohttp.ClientSession() as client:
                async with client.post(local + "/v1/responses",
                    json={"model": cfg.model, "input": "asset", "max_output_tokens": 4096},
                    headers={"Authorization": "Bearer " + token}) as response:
                    assert response.status == 200
                    await response.read()
                await asyncio.wait_for(gateway.handler_finished.get(), 2)
            (record,) = gateway.records("episode")
            assert record["queue_wait_seconds"] == 0.2
            assert 0.05 <= record["provider_request_seconds"] < 0.15
            assert record["elapsed_seconds"] >= 0.25
            assert observed[0]["X-SimReady-Provider-Timeout"] == "0.1"
            assert observed[0]["X-SimReady-Queue-Timeout"] == "1"
            assert float(observed[0]["X-SimReady-Deadline"]) <= controller.get_episode("episode").deadline_at
    asyncio.run(run())


def test_queue_and_provider_timings_survive_failed_attempt_reporting():
    records = [{"status": "completed", "elapsed_seconds": 11,
                "queue_wait_seconds": 8, "provider_request_seconds": 3},
               {"status": "cancelled", "elapsed_seconds": 7,
                "queue_wait_seconds": 5, "provider_request_seconds": 2}]
    requests, statistics = _statistics({"inference_calls_used": 1,
        "inference_attempts_used": 2, "gateway_records": records})
    assert statistics["model_request_seconds"] == 18
    assert statistics["queue_wait_seconds"] == 13
    assert statistics["provider_request_seconds"] == 5
    report = compact_report({"schema": "simready-bench-portable-report-v1", "results": [
        {"condition_id": "timing", "model_requests": requests, "statistics": statistics}
    ]})
    assert report["results"][0]["statistics"]["queue_wait_seconds"] == 13
    assert report["results"][0]["model_requests"][1]["provider_request_seconds"] == 2


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, 1801])
def test_gateway_rejects_unbounded_queue_allowance(bad):
    with pytest.raises(ValueError):
        replace(configuration("http://127.0.0.1:1", 2), queue_wait_timeout_seconds=bad)
