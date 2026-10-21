"""A provider EOF cannot turn an unfinished generation into a completed call."""

import asyncio
from dataclasses import replace
import hashlib
import json

import aiohttp
from aiohttp import web
import pytest

from test_gateway_terminal_response import (
    MODEL,
    CompletedHandlerGateway,
    actor_and_controller,
    configuration,
    serve,
)


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def response_frames(api_style, terminal):
    if api_style == "openai-completions":
        return event(
            {
                "id": "chatcmpl-fixture",
                "model": MODEL,
                "choices": [{"index": 0, "delta": {"content": "asset"}}],
            }
        ) + event(
            {
                "id": "chatcmpl-fixture",
                "choices": [{"index": 0, "delta": {}, "finish_reason": terminal}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ) + b"data: [DONE]\n\n"
    return event(
        {"type": "response.output_text.delta", "delta": "asset"}
    ) + event(
        {
            "type": "response." + (terminal or "in_progress"),
            "response": {
                "id": "resp-fixture",
                "model": MODEL,
                "status": terminal or "in_progress",
                "incomplete_details": (
                    {"reason": "max_output_tokens"}
                    if terminal == "incomplete"
                    else None
                ),
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }
    ) + b"data: [DONE]\n\n"


async def exercise(
    tmp_path, monkeypatch, api_style, body, *, disconnect=False, status=200, output_cap=4096
):
    calls = []
    app = web.Application()
    route = (
        "/v1/chat/completions"
        if api_style == "openai-completions"
        else "/v1/responses"
    )

    async def generate(request):
        calls.append(await request.json())
        response = web.StreamResponse(
            status=status,
            headers={
                "Content-Type": "text/event-stream",
                "X-SimReady-Upstream-Attempts": "1",
                "X-SimReady-Scope-Requests": "1",
            },
        )
        await response.prepare(request)
        # Splits inside JSON and SSE boundaries must preserve terminal detection.
        for offset in range(0, len(body), 17):
            await response.write(body[offset : offset + 17])
        if disconnect:
            request.transport.close()
        else:
            await response.write_eof()
        return response

    async def reserve_cost(self, payload, *args, **kwargs):
        # This loopback transport fixture has no provider pricing. Controller
        # admissions, request ceilings and streaming are real.
        return {
            "request_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        }

    monkeypatch.setattr(CompletedHandlerGateway, "reserve_request_cost", reserve_cost)
    app.router.add_post(route, generate)
    _, controller = actor_and_controller(tmp_path, 1)
    async with serve(app) as upstream:
        config = configuration(upstream, 1)
        config = replace(
            config,
            api_style=api_style,
            max_output_tokens=output_cap,
            native_output_cap=True,
            budget=replace(config.budget, external_spend_cap_verified=True),
        )
        gateway = CompletedHandlerGateway(controller, config)
        token = gateway.issue_episode_token("episode")
        payload = {"model": MODEL, "stream": True}
        if api_style == "openai-completions":
            payload.update(
                messages=[{"role": "user", "content": "asset"}],
                max_completion_tokens=output_cap,
            )
        else:
            payload.update(input="asset", max_output_tokens=output_cap)
        async with serve(gateway.runner()) as local, aiohttp.ClientSession() as client:
            response_error = None
            received = b""
            async with client.post(
                local + route,
                json=payload,
                headers={"Authorization": "Bearer " + token},
            ) as response:
                assert response.status == status, await response.text()
                try:
                    received = await response.read()
                except aiohttp.ClientPayloadError as error:
                    response_error = type(error).__name__
            await asyncio.wait_for(gateway.handler_finished.get(), 2)
            # A failed transport must retain its sole admission. It cannot
            # silently replay or make a second call available.
            async with client.post(
                local + route,
                json=payload,
                headers={"Authorization": "Bearer " + token},
            ) as response:
                assert response.status == 429
                await response.read()
            await asyncio.wait_for(gateway.handler_finished.get(), 2)
        (record,) = gateway.records("episode")
        assert len(calls) == 1
        assert calls[0] == payload
        assert controller.get_episode("episode").inference_calls_used == 1
        assert record["upstream_attempts"] == record["upstream_condition_requests"] == 1
        return record, received, response_error


@pytest.mark.parametrize("api_style", ["openai-completions", "openai-responses"])
def test_clean_http_eof_without_terminal_marker_disconnects_and_retains_evidence(
    tmp_path, monkeypatch, api_style
):
    body = response_frames(api_style, None)
    record, _, error = asyncio.run(exercise(tmp_path, monkeypatch, api_style, body))
    assert error == "ClientPayloadError"
    assert record["status"] == "transport_failed"
    assert record["error_code"] == "upstream_stream_incomplete"
    assert record["observed_upstream_status"] == 200
    assert record["partial_response_bytes"] == len(body)
    assert record["partial_response_sha256"] == hashlib.sha256(body).hexdigest()
    assert record["observed_provider_response_ids"]
    assert record["partial_usage"]
    assert record["partial_usage_terminal"] is False
    assert "usage" not in record and "project_usage_settlement" not in record


@pytest.mark.parametrize("api_style", ["openai-completions", "openai-responses"])
def test_broken_upstream_chunked_transfer_is_not_completed_or_replayed(
    tmp_path, monkeypatch, api_style
):
    body = response_frames(api_style, None)
    record, _, error = asyncio.run(
        exercise(tmp_path, monkeypatch, api_style, body, disconnect=True)
    )
    assert error == "ClientPayloadError"
    assert record["status"] == "transport_failed"
    assert record["error_code"] == "upstream_transport_error"
    assert record["partial_usage_terminal"] is False


@pytest.mark.parametrize(
    "api_style,terminal",
    [
        ("openai-completions", "stop"),
        ("openai-completions", "tool_calls"),
        ("openai-completions", "length"),
        ("openai-responses", "completed"),
        ("openai-responses", "incomplete"),
        ("openai-responses", "failed"),
        ("openai-responses", "cancelled"),
    ],
)
def test_terminal_stream_is_forwarded_unchanged(
    tmp_path, monkeypatch, api_style, terminal
):
    body = response_frames(api_style, terminal)
    record, received, error = asyncio.run(
        exercise(tmp_path, monkeypatch, api_style, body)
    )
    assert error is None and received == body
    assert record["status"] == "completed"
    assert record["response_bytes"] == len(body)
    assert record["response_sha256"] == hashlib.sha256(body).hexdigest()
    assert record["usage"]
    assert "partial_usage_terminal" not in record
    if terminal == "incomplete":
        assert record["provider_response_statuses"] == ["incomplete"]
        assert record["provider_finish_reasons"] == ["max_output_tokens"]


def test_explicit_http_rejection_does_not_require_generation_terminal_marker(
    tmp_path, monkeypatch
):
    body = event({"error": {"message": "Unavailable"}})
    record, received, error = asyncio.run(
        exercise(tmp_path, monkeypatch, "openai-responses", body, status=503)
    )
    assert error is None and received == body
    assert record["status"] == "completed" and record["upstream_status"] == 503
