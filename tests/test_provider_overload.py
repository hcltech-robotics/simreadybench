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

from simready_bench.provider_relay import Relay, RelayState, _retry_after, _rate_limit_reset, _ResponseOutcome
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
def test_rejections_preserve_calls_but_transport_attempts_are_bounded(
    tmp_path, allowance
):
    options = {"maximum_requests": 1} if allowance == "project" else {}
    with relay(tmp_path, [Response(OVERLOAD) for _ in range(4)], **options) as (
        state,
        clock,
        upstream,
        post,
    ):
        limit = 1 if allowance == "condition" else 32
        status, headers, body = post(limit=limit)
        assert (status, body) == (200, OVERLOAD)
        assert headers["X-SimReady-Upstream-Attempts"] == "4"
        assert state.requests == len(upstream.requests) == 4
        assert state.calls == state.scope_calls["condition-a"] == 0
        assert clock.sleeps == [1, 2, 4]
        status, _, body = post(limit=limit)
        assert status == 409
        assert json.loads(body)["error"]["code"] == "relay_allowance_exhausted"
        assert len(upstream.requests) == 4


def test_last_available_call_retries_and_success_consumes_only_one_call(tmp_path):
    with relay(tmp_path, [Response(OVERLOAD), Response(SUCCESS)]) as (
        state, _, upstream, post
    ):
        assert post(limit=1)[2] == SUCCESS
        assert state.calls == state.scope_calls["condition-a"] == 1
        assert state.requests == len(upstream.requests) == 2
        assert post(limit=1)[0] == 409


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


def test_hf_rate_limit_reset_is_used_and_retained(tmp_path):
    quota = '"api";r=0;t=17'
    with relay(tmp_path, [Response(OVERLOAD, 429, {"RateLimit": quota, "RateLimit-Policy": '"api";q=10;w=60'}), Response(SUCCESS)]) as (state, clock, _, post):
        assert post()[2] == SUCCESS
        assert clock.sleeps == [17]
        assert state.events[0]["rate_limit"] == quota
        assert state.events[0]["rate_limit_policy"] == '"api";q=10;w=60'
        assert state.calls == 1


@pytest.mark.parametrize("value,expected", [(None, 0), ('"api";r=1;t=20', 0), ('"api";r=0;t=20', 20), ('"api";r=0;t=20, "other";r=0;t=30', 30), ('"api";r=0;t=NaN', 0)])
def test_rate_limit_reset_only_waits_for_exhausted_valid_quotas(value, expected):
    assert _rate_limit_reset(value) == expected


def test_independent_relays_share_route_permits_until_response_finishes(tmp_path):
    from simready_bench.route_throttle import RouteThrottleServer

    entered, release = threading.Event(), threading.Event()

    class BlockingResponse(Response):
        def read1(self, size):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Fixture response was not released")
            return self.read(size)

    route_key = "a" * 64
    with RouteThrottleServer(tmp_path / "shared.sqlite3", "shared-private-test-token", max_concurrency=1, route_keys=[route_key]) as broker:
        throttle = {"url": broker.url, "token": "shared-private-test-token", "route_key": route_key}
        with relay(tmp_path / "first", [BlockingResponse(SUCCESS)], shared_route_throttle=throttle) as first, relay(tmp_path / "second", [Response(SUCCESS)], shared_route_throttle=throttle) as second:
            results = []
            one = threading.Thread(target=lambda: results.append(first[3]()))
            two = threading.Thread(target=lambda: results.append(second[3]()))
            one.start()
            try:
                assert entered.wait(2)
                two.start()
                assert not threading.Event().wait(0.2)
                assert second[2].requests == []
            finally:
                release.set()
                one.join(5)
                if two.ident is not None:
                    two.join(5)
            assert not one.is_alive() and not two.is_alive()
            assert [result[0] for result in results] == [200, 200]
            assert first[0].calls == second[0].calls == 1
        with sqlite3.connect(tmp_path / "shared.sqlite3") as db:
            assert db.execute("SELECT count(*) FROM leases").fetchone()[0] == 0


def test_shared_route_throttle_publishes_429_cooldown_across_relays(tmp_path, monkeypatch):
    from simready_bench.route_throttle import RouteThrottleServer

    route_key = "b" * 64
    with RouteThrottleServer(tmp_path / "shared.sqlite3", "shared-private-test-token", max_concurrency=1, route_keys=[route_key]) as broker:
        throttle = {"url": broker.url, "token": "shared-private-test-token", "route_key": route_key}
        with relay(tmp_path / "first", [Response(OVERLOAD, 429, {"Retry-After": "60"})], maximum_attempts=1, shared_route_throttle=throttle) as first:
            assert first[3]()[0] == 429
            assert first[0].calls == 0
        with sqlite3.connect(tmp_path / "shared.sqlite3") as db:
            cooldown, failures = db.execute("SELECT cooldown, failures FROM routes WHERE route=?", (route_key,)).fetchone()
            assert cooldown > time.time() and failures == 1
        assert not broker.store.acquire(route_key, "e" * 32)["admitted"]
        monkeypatch.setattr(broker.store, "clock", lambda: cooldown + 1)
        with relay(tmp_path / "second", [Response(SUCCESS)], shared_route_throttle=throttle) as second:
            assert second[3]()[0] == 200
            assert second[0].calls == 1


def test_shared_route_throttle_auth_failure_never_falls_back_to_provider(tmp_path):
    from simready_bench.route_throttle import RouteThrottleServer

    route_key = "c" * 64
    with RouteThrottleServer(tmp_path / "shared.sqlite3", "shared-private-test-token", route_keys=[route_key]) as broker:
        throttle = {"url": broker.url, "token": "incorrect-private-test-token", "route_key": route_key}
        with relay(tmp_path / "relay", [Response(SUCCESS)], shared_route_throttle=throttle) as (state, _, upstream, post):
            assert post()[0] == 502
            assert state.calls == state.requests == 0
            assert upstream.requests == []
            ledger = (tmp_path / "relay" / "ledger.json").read_text()
            assert "incorrect-private-test-token" not in ledger


def test_lost_shared_lease_closes_upstream_and_does_not_replay(tmp_path, monkeypatch):
    from simready_bench.route_throttle import RouteThrottleServer

    closed = threading.Event()

    class LostResponse(Response):
        def read1(self, size):
            if not closed.wait(3):
                raise TimeoutError("Fixture lease was not revoked")
            return self.read(size)

        def close(self):
            super().close()
            closed.set()

    route_key = "d" * 64
    with RouteThrottleServer(tmp_path / "shared.sqlite3", "shared-private-test-token", lease_seconds=1, route_keys=[route_key]) as broker:
        monkeypatch.setattr(broker.store, "heartbeat", lambda *_args: {"renewed": False})
        throttle = {"url": broker.url, "token": "shared-private-test-token", "route_key": route_key}
        with relay(tmp_path / "relay", [LostResponse(SUCCESS)], shared_route_throttle=throttle) as (state, _, upstream, post):
            assert post()[0] == 502
            assert closed.is_set()
            assert state.calls == 0 and state.requests == len(upstream.requests) == 1
            assert upstream.requests[0][1] <= 0.5
            assert state.events[0]["status"] == "transport_error"


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
        assert state.calls == state.scope_calls["condition-a"] == 0


def test_failure_in_later_split_stream_frames_preserves_model_allowance(tmp_path):
    class ChunkedResponse(Response):
        def read1(self, size):
            return self.read(min(size, 7))

    body = b'data: {"type":"response.output_text.delta","delta":"partial"}\r\n\r\ndata: {"type":"response.failed","response":{"error":{"message":"lost"}}}\r\n\r\n'
    with relay(tmp_path, [ChunkedResponse(body, headers={"Content-Type": "text/event-stream"})]) as (state, clock, upstream, post):
        assert post()[2] == body
        assert state.calls == 0 and state.requests == len(upstream.requests) == 1
        assert clock.sleeps == []


@pytest.mark.parametrize("tail", [b'', b'data: [DONE]\n\n'])
def test_clean_stream_eof_without_terminal_response_releases_call(tmp_path, tail):
    body = b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n' + tail
    with relay(tmp_path, [Response(body, headers={"Content-Type": "text/event-stream"})]) as (state, clock, upstream, post):
        assert post()[2] == body
        assert state.calls == 0 and state.requests == len(upstream.requests) == 1
        assert state.events[0]["status"] == "inference_failed"
        assert clock.sleeps == []


@pytest.mark.parametrize("route,value,failed", [
    ("/responses", {"type": "response.cancelled", "response": {"status": "cancelled"}}, True),
    ("/responses", {"response": {"status": "failed"}}, True),
    ("/responses", {"response": {"status": "completed"}}, False),
    ("/chat/completions", {"choices": [{"finish_reason": "error"}]}, True),
    ("/chat/completions", {"choices": [{"finish_reason": "stop"}]}, False),
    ("/chat/completions", {"choices": [{"finish_reason": "length"}]}, False),
    ("/chat/completions", {"choices": [{"delta": {"content": "partial"}}]}, True),
])
def test_stream_terminal_accounting_matches_response_and_chat_protocols(route, value, failed):
    outcome = _ResponseOutcome("text/event-stream", route)
    outcome.feed(b'data: ' + json.dumps(value).encode() + b'\n\n')
    assert outcome.finish() is failed


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
        assert state.calls == state.scope_calls["condition-a"] == 0
        restored = RelayState(config(), directory=tmp_path, now=clock.now)
        assert restored.calls == restored.scope_calls["condition-a"] == 0
        assert restored.requests == restored.scope_requests["condition-a"] == 1


def test_in_flight_call_reservation_prevents_oversubscription_and_releases_once(tmp_path):
    state = RelayState(config(), directory=tmp_path, now=lambda: 1000)
    event = state.admit("/responses", scope="condition-a", limit=1)
    with pytest.raises(ValueError, match="Condition request allowance is exhausted"):
        state.admit("/responses", scope="condition-a", limit=1)
    state.finish(event, "transport_error")
    state.finish(event, "transport_error")
    assert state.calls == 0
    event = state.admit("/responses", scope="condition-a", limit=1)
    state.finish(event, "completed")
    assert state.calls == 1 and state.requests == 2


def test_length_completion_remains_a_productive_model_call(tmp_path):
    body = b'data: {"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}\n\ndata: [DONE]\n\n'
    with relay(tmp_path, [Response(body, headers={"Content-Type": "text/event-stream"})]) as (state, _, _, post):
        assert post()[2] == body
        assert state.calls == 1


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
