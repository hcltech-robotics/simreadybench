"""Transient catalogue loss is bounded without accepting stale price evidence."""

import asyncio

import pytest

from simready_bench.engine import hf_pricing as hf
from simready_bench.engine.request_cost import CostDenied
from test_hf_cost import MODEL, catalog


def run_probe(monkeypatch, responses, *, deadline=1000):
    clock = [100.0]
    calls, waits = [], []
    remaining = iter(responses)

    async def fetch(model, *, deadline_at):
        calls.append((model, deadline_at))
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        return hf.validate_catalog_route(response, model)

    async def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(hf.time, "time", lambda: clock[0])
    monkeypatch.setattr(hf, "_fetch_hf_route_once", fetch)
    monkeypatch.setattr(hf.asyncio, "sleep", sleep)
    return calls, waits, lambda: asyncio.run(hf.fetch_hf_route(MODEL, deadline_at=deadline))


@pytest.mark.parametrize("code", ["hf_pricing_route_unavailable", "hf_pricing_metadata_unavailable"])
def test_transient_failure_requires_new_matching_catalogue(monkeypatch, code):
    calls, waits, run = run_probe(monkeypatch, [CostDenied(code), catalog()])
    route = run()
    assert route.model == MODEL
    assert route.pricing == hf.HF_PRICING[MODEL]
    assert route.verified_at == 105
    assert calls == [(MODEL, 220), (MODEL, 220)]
    assert waits == [5]


def test_missing_route_recovers_only_after_exact_provider_returns(monkeypatch):
    missing = catalog()
    missing["data"][0]["providers"] = []
    calls, waits, run = run_probe(monkeypatch, [missing, missing, catalog()])
    assert run().model == MODEL
    assert len(calls) == 3
    assert waits == [5, 10]


def test_persistent_missing_route_has_five_attempts_and_no_fallback(monkeypatch):
    calls, waits, run = run_probe(monkeypatch, [{"data": []}] * 5)
    with pytest.raises(CostDenied, match="hf_pricing_route_unavailable"):
        run()
    assert len(calls) == 5
    assert waits == [5, 10, 20, 40]


@pytest.mark.parametrize("code", ["hf_pricing_metadata_changed", "hf_pricing_metadata_invalid", "hf_pricing_response_limit"])
def test_changed_or_invalid_prices_are_not_retried_or_waived(monkeypatch, code):
    calls, waits, run = run_probe(monkeypatch, [CostDenied(code), catalog()])
    with pytest.raises(CostDenied, match=code):
        run()
    assert len(calls) == 1
    assert waits == []


def test_caller_deadline_prevents_retry_sleep(monkeypatch):
    calls, waits, run = run_probe(monkeypatch, [{"data": []}], deadline=104)
    with pytest.raises(CostDenied, match="hf_pricing_route_unavailable"):
        run()
    assert calls == [(MODEL, 104)]
    assert waits == []


def test_expired_deadline_makes_no_metadata_request(monkeypatch):
    calls, waits, run = run_probe(monkeypatch, [catalog()], deadline=100)
    with pytest.raises(CostDenied, match="hf_pricing_deadline_exceeded"):
        run()
    assert calls == waits == []


def test_http_catalogue_recovery_preserves_exact_prices_and_sends_no_credentials(monkeypatch):
    from aiohttp import web
    import time

    async def exercise():
        requests, waits = [], []

        async def models(request):
            requests.append(dict(request.headers))
            return web.json_response({"data": []} if len(requests) == 1 else catalog())

        async def sleep(seconds):
            waits.append(seconds)

        app = web.Application()
        app.router.add_get("/v1/models", models)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(hf, "HF_CATALOG_URL", f"http://127.0.0.1:{port}/v1/models")
        monkeypatch.setattr(hf.asyncio, "sleep", sleep)
        try:
            result = await hf.fetch_hf_route(MODEL, deadline_at=time.time() + 30)
            assert result.model == MODEL
            assert result.pricing == hf.HF_PRICING[MODEL]
            assert len(requests) == 2
            assert waits == [5]
            assert all("Authorization" not in headers for headers in requests)
        finally:
            await runner.cleanup()

    asyncio.run(exercise())
