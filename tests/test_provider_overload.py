import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import format_datetime
import http.server
import io
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request

import aiohttp
from aiohttp import web
import pytest

from simready_bench.provider_relay import Relay, RelayState, _retry_after
from test_gateway_terminal_response import (
    CompletedHandlerGateway,
    actor_and_controller,
    configuration,
    serve,
)


OVERLOAD = b'{"error":{"code":"server_error","message":"Our servers are currently overloaded. Please try again later."}}'
SUCCESS = b'{"model":"gpt-6.1-sol","output":[{"type":"message","content":[{"text":"asset authored"}]}],"usage":{"input_tokens":20,"output_tokens":5}}'
RAW = b'{"model":"gpt-6.1-sol","input":"reference","stream":false}'


class Clock:
    def __init__(self):
        self.value = 1000.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class Response(io.BytesIO):
    def __init__(self, raw, status=200, headers=None):
        super().__init__(raw)
        self.status = status
        self.headers = {"Content-Type": "application/json", **(headers or {})}


class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        selected = self.responses.pop(0)
        if isinstance(selected, Exception):
            raise selected
        return selected


def config(**options):
    return {
        "model": "gpt-6.1-sol",
        "marker": "local-delegated-fixture",
        "api_key": "local-upstream-fixture",
        "base_url": "https://api.openai.com/v1",
        "maximum_requests": 64,
        "maximum_condition_requests": 32,
        "expires_at": 1500.0,
        **options,
    }


@contextmanager
def relay(tmp_path, responses, **options):
    clock, opener = Clock(), Opener(responses)
    jitter = options.pop("jitter", lambda value: value)
    state = RelayState(
        config(**options),
        directory=tmp_path,
        opener=opener,
        now=clock.now,
        sleep=clock.sleep,
        jitter=jitter,
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(*, scope="condition-a", limit=32, deadline=1400, raw=RAW):
        headers = {
            "Authorization": "Bearer local-delegated-fixture",
            "Content-Type": "application/json",
        }
        if scope is not None:
            headers.update(
                {
                    "X-SimReady-Condition": scope,
                    "X-SimReady-Request-Limit": str(limit),
                    "X-SimReady-Deadline": str(deadline),
                }
            )
        request = urllib.request.Request(
            "http://127.0.0.1:" + str(server.server_port) + "/v1/responses",
            data=raw,
            headers=headers,
        )
        try:
            response = client.open(request, timeout=2)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, dict(response.headers), response.read()

    try:
        yield state, clock, opener, post
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_200_overload_retries_without_exposing_error_or_rewriting_payload(
    tmp_path,
):
    with relay(tmp_path, [Response(OVERLOAD), Response(SUCCESS)]) as (
        state,
        clock,
        upstream,
        post,
    ):
        status, headers, body = post()
        assert (status, body) == (200, SUCCESS)
        assert headers["X-SimReady-Upstream-Attempts"] == "2"
        assert headers["X-SimReady-Scope-Requests"] == "2"
        assert state.requests == state.scope_requests["condition-a"] == 2
        assert clock.sleeps == [1]
        assert all(request.data == RAW for request, _ in upstream.requests)
        assert all(
            request.get_header("Authorization") == "Bearer local-upstream-fixture"
            for request, _ in upstream.requests
        )
        assert [event["status"] for event in state.events] == [
            "retryable_rejection",
            "completed",
        ]
        assert "local-upstream-fixture" not in (tmp_path / "ledger.json").read_text()


def test_sse_overload_after_control_frames_retries_before_generated_output(tmp_path):
    rejected = b'event: response.created\ndata: {"type":"response.created","response":{"output":[]}}\n\nevent: error\ndata: {"type":"error","message":"Our servers are currently overloaded. Please try again later."}\n\ndata: [DONE]\n\n'
    success = b'data: {"type":"response.output_text.delta","delta":"asset"}\n\ndata: [DONE]\n\n'
    headers = {"Content-Type": "text/event-stream"}
    with relay(
        tmp_path,
        [Response(rejected, headers=headers), Response(success, headers=headers)],
    ) as (state, _, _, post):
        assert post()[2] == success
        assert state.requests == 2


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_explicit_transient_http_rejection_honours_retry_after(tmp_path, status):
    with relay(
        tmp_path,
        [
            Response(b'{"error":"unavailable"}', status, {"Retry-After": "3"}),
            Response(SUCCESS),
        ],
    ) as (state, clock, _, post):
        assert post()[2] == SUCCESS
        assert state.requests == 2 and clock.sleeps == [3]


def test_exponential_backoff_is_jittered_and_attempts_are_bounded(tmp_path):
    replies = [Response(OVERLOAD) for _ in range(3)] + [Response(SUCCESS)]
    with relay(tmp_path, replies, jitter=lambda value: value * 1.25) as (
        state,
        clock,
        _,
        post,
    ):
        assert post()[2] == SUCCESS
        assert state.requests == 4
        assert clock.sleeps == [1.25, 2.5, 5]


@pytest.mark.parametrize("allowance", ["condition", "project"])
def test_retries_stop_at_actual_condition_or_project_request_ceiling(
    tmp_path, allowance
):
    options = {"maximum_requests": 2} if allowance == "project" else {}
    with relay(tmp_path, [Response(OVERLOAD), Response(OVERLOAD)], **options) as (
        state,
        clock,
        upstream,
        post,
    ):
        status, headers, body = post(limit=2 if allowance == "condition" else 32)
        assert (status, body) == (200, OVERLOAD)
        assert headers["X-SimReady-Upstream-Attempts"] == "2"
        assert state.requests == len(upstream.requests) == 2
        assert clock.sleeps == [1]


def test_retry_after_beyond_deadline_does_not_dispatch_again(tmp_path):
    with relay(tmp_path, [Response(OVERLOAD, headers={"Retry-After": "15"})]) as (
        state,
        clock,
        upstream,
        post,
    ):
        assert post(deadline=1005)[2] == OVERLOAD
        assert state.requests == len(upstream.requests) == 1 and clock.sleeps == []


def test_elapsed_retry_window_stops_before_another_attempt(tmp_path):
    with relay(
        tmp_path, [Response(OVERLOAD), Response(OVERLOAD)], maximum_retry_seconds=2
    ) as (state, clock, upstream, post):
        assert post()[2] == OVERLOAD
        assert state.requests == len(upstream.requests) == 2 and clock.sleeps == [1]


def test_retry_after_supports_http_dates_without_shortening_the_wait():
    value = format_datetime(datetime.fromtimestamp(1017, timezone.utc), usegmt=True)
    assert _retry_after(value, 1000) == 17
    assert _retry_after("NaN", 1000) == _retry_after("bad", 1000) == 0


@pytest.mark.parametrize(
    "status,body",
    [
        (400, OVERLOAD),
        (401, OVERLOAD),
        (200, b'{"error":{"message":"Bad input"}}'),
        (503, SUCCESS),
    ],
)
def test_nonretryable_or_generated_responses_are_forwarded_once(tmp_path, status, body):
    with relay(tmp_path, [Response(body, status)]) as (state, clock, upstream, post):
        assert post()[::2] == (status, body)
        assert state.requests == len(upstream.requests) == 1 and clock.sleeps == []


def test_stream_error_after_generation_is_never_replayed(tmp_path):
    body = b'data: {"type":"response.output_text.delta","delta":"authored"}\n\ndata: {"type":"error","message":"Our servers are currently overloaded. Please try again later."}\n\n'
    with relay(
        tmp_path, [Response(body, headers={"Content-Type": "text/event-stream"})]
    ) as (state, clock, upstream, post):
        assert post()[2] == body
        assert state.requests == len(upstream.requests) == 1 and clock.sleeps == []


@pytest.mark.parametrize(
    "status,body,headers",
    [
        (200, b'{"error":{"code":"overloaded"},"usage":{"input_tokens":1}}', {}),
        (
            503,
            b'data: {"type":"response.output_text.delta","delta":"asset"}\n\n',
            {"Content-Type": "text/event-stream"},
        ),
    ],
)
def test_reported_usage_or_stream_output_prevents_retry(
    tmp_path, status, body, headers
):
    with relay(tmp_path, [Response(body, status, headers)]) as (
        state,
        clock,
        upstream,
        post,
    ):
        assert post()[2] == body
        assert state.requests == len(upstream.requests) == 1 and clock.sleeps == []


def test_ambiguous_transport_failure_is_not_replayed(tmp_path):
    with relay(tmp_path, [urllib.error.URLError("response state unknown")]) as (
        state,
        clock,
        upstream,
        post,
    ):
        assert post()[0] == 502
        assert state.requests == len(upstream.requests) == 1 and clock.sleeps == []
        assert state.events[0]["status"] == "transport_error"


def test_existing_ledger_keeps_actual_attempts_and_refuses_changed_allowance(tmp_path):
    state = RelayState(config(), directory=tmp_path, now=lambda: 1000)
    state.admit("/responses", scope="condition-a", limit=2)
    restored = RelayState(config(), directory=tmp_path, now=lambda: 1000)
    assert restored.requests == restored.scope_requests["condition-a"] == 1
    restored.admit("/responses", scope="condition-a", limit=2)
    with pytest.raises(ValueError, match="Condition request allowance is exhausted"):
        restored.admit("/responses", scope="condition-a", limit=2)
    with pytest.raises(ValueError, match="cannot change"):
        RelayState(config(maximum_requests=128), directory=tmp_path, now=lambda: 1000)


def test_conditions_share_the_global_cap_without_borrowing_each_others_requests(
    tmp_path,
):
    state = RelayState(config(maximum_requests=3), directory=tmp_path, now=lambda: 1000)
    for _ in range(2):
        state.admit("/responses", scope="condition-a", limit=2)
    with pytest.raises(ValueError, match="Condition request allowance is exhausted"):
        state.admit("/responses", scope="condition-a", limit=2)
    state.admit("/responses", scope="condition-b", limit=2)
    with pytest.raises(ValueError, match="Model request allowance is exhausted"):
        state.admit("/responses", scope="condition-b", limit=2)


def test_gateway_owns_retry_headers_and_retains_attempt_metadata(tmp_path):
    async def run():
        observed = []
        app = web.Application()

        async def counter(request):
            return web.json_response(
                {"object": "response.input_tokens", "input_tokens": 100}
            )

        async def generate(request):
            observed.append(dict(request.headers))
            return web.json_response(
                {
                    "model": "gpt-6-astra",
                    "output": [],
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
                headers={
                    "X-SimReady-Upstream-Attempts": "3",
                    "X-SimReady-Scope-Requests": "7",
                },
            )

        app.router.add_post("/v1/responses/input_tokens", counter)
        app.router.add_post("/v1/responses", generate)
        _, controller = actor_and_controller(tmp_path, 8)
        async with serve(app) as upstream:
            gateway = CompletedHandlerGateway(controller, configuration(upstream, 16))
            token = gateway.issue_episode_token("episode")
            async with (
                serve(gateway.runner()) as local,
                aiohttp.ClientSession() as client,
            ):
                payload = {
                    "model": "gpt-6-astra",
                    "input": "asset",
                    "max_output_tokens": 4096,
                }
                async with client.post(
                    local + "/v1/responses",
                    json=payload,
                    headers={"Authorization": "Bearer " + token},
                ) as response:
                    assert response.status == 200
                    await response.read()
                await asyncio.wait_for(gateway.handler_finished.get(), 2)
                async with client.post(
                    local + "/v1/responses",
                    json=payload,
                    headers={
                        "Authorization": "Bearer " + token,
                        "X-SimReady-Condition": "forged",
                    },
                ) as response:
                    assert response.status == 400
                    await response.read()
                await asyncio.wait_for(gateway.handler_finished.get(), 2)
            assert len(observed) == 1
            assert observed[0]["X-SimReady-Condition"] == "local-regression-authority"
            assert observed[0]["X-SimReady-Request-Limit"] == "8"
            assert (
                float(observed[0]["X-SimReady-Deadline"])
                <= controller.get_episode("episode").deadline_at
            )
            (record,) = gateway.records("episode")
            assert (
                record["upstream_attempts"] == 3
                and record["upstream_condition_requests"] == 7
            )

    asyncio.run(run())


def test_local_https_gateway_reuses_backoff_and_single_spend_reservation(
    tmp_path, monkeypatch
):
    from simready_bench import provider_relay
    from simready_bench.engine import request_cost

    success = json.dumps(
        {
            "model": "gpt-6-astra",
            "output": [],
            "usage": {"input_tokens": 100, "output_tokens": 1, "total_tokens": 101},
        }
    ).encode()
    opener = Opener([Response(OVERLOAD), Response(success)])
    states, delays, counted = [], [], []

    def state_factory(config, *, directory):
        state = RelayState(
            config,
            directory=directory,
            opener=opener,
            sleep=delays.append,
            jitter=lambda value: value,
        )
        states.append(state)
        return state

    async def counter(config, payload, **bounds):
        counted.append((config, bounds))
        return request_cost.InputTokenEvidence(
            config.model,
            100,
            request_cost.sha(payload),
            request_cost.sha(
                {
                    key: value
                    for key, value in payload.items()
                    if key in request_cost.COUNT_KEYS
                }
            ),
            config.upstream_origin,
            time.time(),
        )

    monkeypatch.setattr(provider_relay, "RelayState", state_factory)
    monkeypatch.setattr(request_cost, "count_input_tokens", counter)

    async def run():
        _, controller = actor_and_controller(tmp_path, 4)
        cfg = configuration("https://api.openai.com", 4)
        gateway = CompletedHandlerGateway(controller, cfg)
        token = gateway.issue_episode_token("episode")
        async with serve(gateway.runner()) as local, aiohttp.ClientSession() as client:
            async with client.post(
                local + "/v1/responses",
                json={"model": cfg.model, "input": "asset", "max_output_tokens": 4096},
                headers={"Authorization": "Bearer " + token},
            ) as response:
                assert response.status == 200
                assert await response.read() == success
            await asyncio.wait_for(gateway.handler_finished.get(), 2)
            assert gateway._provider_relay_thread.is_alive()
        assert gateway._provider_relay_server is gateway._provider_relay_thread is None
        assert len(counted) == 1
        assert len(opener.requests) == 2 and delays == [1]
        assert states[0].requests == 2
        assert controller.get_episode("episode").inference_calls_used == 1
        (record,) = gateway.records("episode")
        assert record["upstream_attempts"] == record["upstream_condition_requests"] == 2
        assert record["project_usage_settlement"]["settled_upper_micro_usd"] > 0
        with sqlite3.connect(controller.state_dir / "request-cost.sqlite3") as db:
            assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 1
            assert (
                db.execute("SELECT COUNT(*) FROM cost_settlements").fetchone()[0] == 1
            )
        for request, _ in opener.requests:
            assert request.full_url == "https://api.openai.com/v1/responses"
            assert (
                request.get_header("Authorization") == "Bearer " + cfg.upstream_api_key
            )
            assert not any(
                key.lower().startswith("x-simready-")
                for key, _ in request.header_items()
            )
        restored = CompletedHandlerGateway(controller, cfg)
        async with serve(restored.runner()):
            assert states[-1].requests == 2

    asyncio.run(run())
