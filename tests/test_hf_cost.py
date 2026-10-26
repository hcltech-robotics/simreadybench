"""No inference: pricing guards and durable reservations against public fixtures."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import time

import pytest
from simready_bench.engine.inference_gateway import GatewayConfig, ProjectBudget
from simready_bench.engine import request_cost as cost

MODEL = "Qwen/Qwen3.8-27B:deepinfra"
ORIGIN = "https://router.huggingface.co"


def config(cap=1):
    budget = ProjectBudget(
        "simready-bench",
        "hf-test",
        True,
        ORIGIN,
        MODEL,
        time.time() + 120,
        20,
        cap,
        pricing_status="known",
    )
    return GatewayConfig(
        ORIGIN + "/v1",
        MODEL,
        "fixture-only-no-credential",
        api_style="openai-completions",
        budget=budget,
    )


def catalog():
    return {
        "data": [
            {
                "id": "Qwen/Qwen3.8-27B",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "providers": [
                    {
                        "provider": "deepinfra",
                        "status": "live",
                        "context_length": 262144,
                        "pricing": {"input": 0.20, "output": 2.50},
                        "supports_tools": True,
                    }
                ],
            }
        ]
    }


def evidence(payload):
    from simready_bench.engine.hf_pricing import (
        validate_catalog_route,
        HFContextEvidence,
    )

    route = validate_catalog_route(catalog(), MODEL)
    return HFContextEvidence(MODEL, cost.sha(payload), ORIGIN, route, time.time())


def native_context_fixture(model, *, native=True):
    from simready_bench.engine.hf_pricing import HF_PRICING, validate_catalog_route

    price = HF_PRICING[model]
    cfg = config(cap=10)
    cfg = replace(cfg, model=model, budget=replace(cfg.budget, model=model),
                  native_output_limits=native, max_input_tokens=price.context_length,
                  max_output_tokens=None if native else 4096)
    doc = catalog()
    row = doc["data"][0]
    row["id"] = model.rsplit(":", 1)[0]
    row["providers"][0].update(provider=model.rsplit(":", 1)[1], context_length=price.context_length,
                               pricing={"input": price.input_per_million_usd, "output": price.output_per_million_usd})
    return cfg, validate_catalog_route(doc, model)


@pytest.mark.parametrize("model", ["zai-org/GLM-5.3-Flash:baseten", "meta-models/Muse-Spark-Open-816B-A42B:baseten"])
@pytest.mark.parametrize("native_output", [False, True])
def test_native_million_token_context_is_verified_and_reserved(tmp_path, monkeypatch, model, native_output):
    from simready_bench.engine import hf_pricing as hf

    cfg, route = native_context_fixture(model, native=native_output)
    payload = {"model": model, "messages": []}
    if not native_output:
        payload["max_tokens"] = 4096

    async def fetch(*args, **kwargs):
        return route

    monkeypatch.setattr(hf, "fetch_hf_route", fetch)
    verified = asyncio.run(cost.verify_hf_context_bound(cfg, payload, maximum_input_tokens=1048576, deadline_at=time.time() + 5))
    reservation = cost.CostLedger(tmp_path).reserve(budget=cfg.budget, payload=payload, evidence=verified,
        request_id="native-million", purpose="actor", maximum_input_tokens=1048576, native_output_limits=native_output)
    assert reservation["declared_maximum_input_tokens"] == 1048576
    assert reservation["input_token_upper_bound"] == 1048576
    assert reservation["input_count_verified"] is False
    assert reservation["actual_cost_usd"] is None
    if model == "zai-org/GLM-5.3-Flash:baseten" and not native_output:
        assert reservation["reserved_upper_cost_micro_usd"] == 159335


@pytest.mark.parametrize("native,limit", [(True, 262144), (True, 1048575), (True, 1048577), (True, True), (True, None), (False, 262144), (False, 1048577)])
def test_context_verification_and_reservation_retain_exact_native_and_bounded_caps(tmp_path, monkeypatch, native, limit):
    from simready_bench.engine import hf_pricing as hf

    model = "zai-org/GLM-5.3-Flash:baseten"
    cfg, route = native_context_fixture(model, native=native)
    payload = {"model": model, "messages": [], "max_tokens": 4096}
    fetched = []

    async def fetch(*args, **kwargs):
        fetched.append(True)
        return route

    monkeypatch.setattr(hf, "fetch_hf_route", fetch)
    with pytest.raises(cost.CostDenied, match="hf_input_limit_unverifiable"):
        asyncio.run(cost.verify_hf_context_bound(cfg, payload, maximum_input_tokens=limit, deadline_at=time.time() + 5))
    assert fetched == []
    verified = hf.HFContextEvidence(model, cost.sha(payload), ORIGIN, route, time.time())
    with pytest.raises(cost.CostDenied, match="hf_input_limit_unverifiable"):
        cost.CostLedger(tmp_path).reserve(budget=cfg.budget, payload=payload, evidence=verified,
            request_id="invalid-bound", purpose="actor", maximum_input_tokens=limit, native_output_limits=native)


def test_million_token_native_reservation_rejects_changed_route_pricing(tmp_path):
    from simready_bench.engine import hf_pricing as hf

    model = "zai-org/GLM-5.3-Flash:baseten"
    cfg, route = native_context_fixture(model)
    payload = {"model": model, "messages": [], "max_tokens": 4096}
    changed = replace(route, pricing=replace(route.pricing, input_per_million_usd="0.16"))
    verified = hf.HFContextEvidence(model, cost.sha(payload), ORIGIN, changed, time.time())
    with pytest.raises(cost.CostDenied, match="hf_context_evidence_required"):
        cost.CostLedger(tmp_path).reserve(budget=cfg.budget, payload=payload, evidence=verified,
            request_id="changed-price", purpose="actor", maximum_input_tokens=1048576, native_output_limits=True)


def test_local_glm_capped_gateway_verifies_and_reserves_its_native_input_context(tmp_path, monkeypatch):
    from dataclasses import asdict
    from simready_bench.local import _gateway
    from simready_bench.engine import hf_pricing as hf

    model = "zai-org/GLM-5.3-Flash:baseten"
    original, route = native_context_fixture(model, native=False)
    condition = {"condition_id": "fixture-condition", "authoring_protocol": "authoring-64-bounded-output-4096-v1-tools-bounded-v1-continue-length-v1",
                 "model": {"model_id": model, "provider_ref": "selected", "adapter": "openai-completions"}, "harness": {"adapter": "pi"},
                 "settings": {"generation": {"max_output_tokens": 4096}, "budgets": {"maximum_spend_usd": 10, "inference_requests": 64,
                    "max_input_tokens": 1048576, "max_output_tokens": 4096, "agent_seconds": 1800}}}
    resolved = {"operator_policy": {"providers": {"selected": {"kind": "huggingface", "api_key_env": "LOCAL_TEST_HF_KEY",
                    "base_url": ORIGIN + "/v1", "project_budget": asdict(original.budget)}}}}
    monkeypatch.setenv("LOCAL_TEST_HF_KEY", "fixture-only-no-credential")
    cfg = _gateway(condition, resolved)
    assert cfg.max_input_tokens == 1048576 and cfg.max_output_tokens == 4096
    assert cfg.native_output_limits is False
    payload = {"model": model, "messages": [], "max_tokens": 4096}

    async def fetch(*args, **kwargs):
        return route

    monkeypatch.setattr(hf, "fetch_hf_route", fetch)
    verified = asyncio.run(cost.verify_hf_context_bound(cfg, payload, maximum_input_tokens=cfg.max_input_tokens, deadline_at=time.time() + 5))
    reserved = cost.CostLedger(tmp_path).reserve(budget=cfg.budget, payload=payload, evidence=verified,
        request_id="local-capped", purpose="actor", maximum_input_tokens=cfg.max_input_tokens, native_output_limits=cfg.native_output_limits)
    assert reserved["input_token_upper_bound"] == 1048576 and reserved["max_output_tokens"] == 4096
    assert reserved["reserved_upper_cost_micro_usd"] == 159335


def test_sol_uses_current_higher_cache_write_rate():
    assert cost.upper_cost_micro_usd(64000, 4096, model="gpt-6.1-sol") == 200960


def test_hf_has_fixed_provider_and_never_claims_a_verified_token_count(tmp_path):
    cfg = config()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 4096,
    }
    ev = evidence(payload)
    result = cost.CostLedger(tmp_path).reserve(
        budget=cfg.budget,
        payload=payload,
        evidence=ev,
        request_id="one",
        purpose="actor",
        maximum_input_tokens=262144,
    )
    assert result["input_tokens"] is None
    assert result["input_token_upper_bound"] == 262144
    assert result["input_count_verified"] is False
    assert result["reserved_upper_cost_micro_usd"] == 62669
    assert result["actual_cost_usd"] is None
    assert result["reservation_basis"] == "hf-provider-context-capacity"
    assert cfg.budget.external_spend_cap_verified is False


@pytest.mark.parametrize("limit", [None, 64000, 256000, True])
def test_context_reservation_cannot_silently_ignore_declared_input_cap(tmp_path, limit):
    payload = {"model": MODEL, "messages": [], "max_tokens": 4096}
    with pytest.raises(cost.CostDenied, match="hf_input_limit_unverifiable"):
        cost.CostLedger(tmp_path).reserve(
            budget=config().budget,
            payload=payload,
            evidence=evidence(payload),
            request_id="one",
            purpose="actor",
            maximum_input_tokens=limit,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "price",
        "context",
        "status",
        "tools",
        "image",
        "missing",
        "ambiguous",
        "extra-price",
    ],
)
def test_public_catalog_route_fails_closed_when_metadata_changes(mutation):
    from simready_bench.engine.hf_pricing import validate_catalog_route

    doc = catalog()
    row = doc["data"][0]
    provider = row["providers"][0]
    if mutation == "price":
        provider["pricing"]["output"] = 2.51
    if mutation == "context":
        provider["context_length"] = 262145
    if mutation == "status":
        provider["status"] = "staging"
    if mutation == "tools":
        provider["supports_tools"] = False
    if mutation == "image":
        row["architecture"]["input_modalities"] = ["text"]
    if mutation == "missing":
        row["providers"] = []
    if mutation == "ambiguous":
        row["providers"].append(deepcopy(provider))
    if mutation == "extra-price":
        provider["pricing"]["image"] = 1
    with pytest.raises(cost.CostDenied):
        validate_catalog_route(doc, MODEL)


def test_unknown_or_unpinned_routes_never_qualify():
    assert cost.supports_application_spend_cap(config())
    for model in (
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-27B:cheapest",
        MODEL + ":deepinfra",
    ):
        cfg = config()
        assert not cost.supports_application_spend_cap(
            replace(cfg, model=model, budget=replace(cfg.budget, model=model))
        )
    assert not cost.supports_application_spend_cap(
        replace(config(), upstream_base_url="https://elsewhere.example/v1")
    )
    assert not cost.supports_application_spend_cap(
        replace(config(), api_style="openai-responses")
    )


def test_glm_flash_reserves_full_context_and_pins_its_provider():
    from simready_bench.benchmark import native_run_allowance
    from simready_bench.engine.hf_pricing import (
        hf_upper_cost_micro_usd, validate_catalog_route,
    )

    model = "zai-org/GLM-5.3-Flash:baseten"
    doc = catalog()
    row = doc["data"][0]
    row["id"] = "zai-org/GLM-5.3-Flash"
    provider = row["providers"][0]
    provider.update(provider="baseten", context_length=1048576,
                    pricing={"input": 0.15, "output": 0.50})
    assert validate_catalog_route(doc, model).model == model
    assert hf_upper_cost_micro_usd(model, 8192, native_output_limits=True) == 161383
    assert native_run_allowance(model, provider="huggingface", conditions=3,
                                authoring_turns=64, max_output_tokens=8192) == 30.985536
    provider["provider"] = "deepinfra"
    with pytest.raises(cost.CostDenied, match="hf_pricing_route_unavailable"):
        validate_catalog_route(doc, model)


def test_context_evidence_stale_or_rebound_is_rejected(tmp_path):
    cfg = config()
    payload = {"model": MODEL, "messages": [], "max_tokens": 4096}
    ev = evidence(payload)
    for invalid in (
        replace(ev, request_sha256="0" * 64),
        replace(ev, verified_at=time.time() - 31),
        replace(ev, verified_at=float("nan")),
        replace(ev, upstream_origin="https://elsewhere.example"),
    ):
        with pytest.raises(cost.CostDenied):
            cost.CostLedger(tmp_path).reserve(
                budget=cfg.budget,
                payload=payload,
                evidence=invalid,
                request_id="bad",
                purpose="actor",
                maximum_input_tokens=262144,
            )


def test_hf_shared_cap_is_atomic_and_attempts_are_not_refunded(tmp_path):
    cfg = config(cap=0.125338)
    payload = {"model": MODEL, "messages": [], "max_tokens": 4096}
    ev = evidence(payload)
    ledger = cost.CostLedger(tmp_path)

    def reserve(i):
        try:
            return ledger.reserve(
                budget=cfg.budget,
                payload=payload,
                evidence=ev,
                request_id=str(i),
                purpose="actor",
                maximum_input_tokens=262144,
            )
        except cost.CostDenied as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(16)))
    assert sum(isinstance(v, dict) for v in results) == 2
    assert results.count("project_spend_budget_exhausted") == 14


@pytest.mark.parametrize(
    "change",
    [
        {"max_tokens": True},
        {"max_tokens": 262145},
        {"n": 2},
        {"max_completion_tokens": 4096},
    ],
)
def test_unbounded_or_ambiguous_generation_is_rejected(tmp_path, change):
    payload = {"model": MODEL, "messages": [], "max_tokens": 4096, **change}
    with pytest.raises(cost.CostDenied):
        cost.CostLedger(tmp_path).reserve(
            budget=config().budget,
            payload=payload,
            evidence=evidence(payload),
            request_id="bad",
            purpose="actor",
            maximum_input_tokens=262144,
        )


def test_explicit_common_input_policy_extends_openai_without_changing_legacy_default(
    tmp_path,
):
    from simready_bench.engine.request_cost import InputTokenEvidence

    cfg = config()
    approved = replace(
        cfg.budget,
        model="gpt-6-astra",
        upstream_origin="https://api.openai.com",
        maximum_spend_usd=10,
    )
    payload = {"model": "gpt-6-astra", "input": "test", "max_output_tokens": 4096}
    ev = InputTokenEvidence(
        approved.model,
        262144,
        cost.sha(payload),
        "a" * 64,
        approved.upstream_origin,
        time.time(),
    )
    with pytest.raises(cost.CostDenied):
        cost.CostLedger(tmp_path).reserve(
            budget=approved,
            payload=payload,
            evidence=ev,
            request_id="legacy",
            purpose="actor",
        )
    result = cost.CostLedger(tmp_path).reserve(
        budget=approved,
        payload=payload,
        evidence=ev,
        request_id="common-policy",
        purpose="actor",
        maximum_input_tokens=262144,
    )
    assert result["reserved_upper_cost_micro_usd"] == 3481600
    assert result["input_tokens"] == 262144
    assert (
        cost.upper_cost_micro_usd(
            262144, 4096, model="gpt-6-astra", maximum_input_tokens=262144
        )
        == 3481600
    )


def test_public_catalog_is_revalidated_each_time_without_authorization(monkeypatch):
    from aiohttp import web
    from simready_bench.engine import hf_pricing as hf

    async def run():
        requests = []
        doc = catalog()

        async def models(request):
            requests.append(dict(request.headers))
            return web.json_response(doc)

        app = web.Application()
        app.router.add_get("/v1/models", models)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = (
            "http://127.0.0.1:"
            + str(site._server.sockets[0].getsockname()[1])
            + "/v1/models"
        )
        monkeypatch.setattr(hf, "HF_CATALOG_URL", url)
        try:
            payload = {"model": MODEL, "messages": [], "max_tokens": 4096}
            ev = await cost.verify_hf_context_bound(
                config(),
                payload,
                maximum_input_tokens=262144,
                deadline_at=time.time() + 5,
            )
            assert ev.route.pricing.context_length == 262144
            doc["data"][0]["providers"][0]["pricing"]["input"] = 0.21
            with pytest.raises(cost.CostDenied, match="hf_pricing_metadata_changed"):
                await cost.verify_hf_context_bound(
                    config(),
                    payload,
                    maximum_input_tokens=262144,
                    deadline_at=time.time() + 5,
                )
            assert len(requests) == 2
            assert all("Authorization" not in h for h in requests)
            with pytest.raises(cost.CostDenied, match="hf_input_limit_unverifiable"):
                await cost.verify_hf_context_bound(
                    config(),
                    payload,
                    maximum_input_tokens=64000,
                    deadline_at=time.time() + 5,
                )
            assert len(requests) == 2
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_hf_public_catalog_redirects_are_not_followed(monkeypatch):
    from aiohttp import web
    from simready_bench.engine import hf_pricing as hf

    # A six-second retry cannot fit the five-second admission deadline. Keep
    # this HTTP redirect check independent of randomized retry scheduling.
    monkeypatch.setattr(hf.random, "uniform", lambda low, high: high)

    async def run():
        redirects = []
        followed = []

        async def redirect(request):
            redirects.append(True)
            raise web.HTTPFound("/destination")

        async def destination(request):
            followed.append(True)
            return web.json_response(catalog())

        app = web.Application()
        app.router.add_get("/v1/models", redirect)
        app.router.add_get("/destination", destination)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        monkeypatch.setattr(
            hf,
            "HF_CATALOG_URL",
            "http://127.0.0.1:"
            + str(site._server.sockets[0].getsockname()[1])
            + "/v1/models",
        )
        try:
            with pytest.raises(
                cost.CostDenied, match="hf_pricing_metadata_unavailable"
            ):
                await hf.fetch_hf_route(MODEL, deadline_at=time.time() + 5)
            assert redirects == [True]
            assert not followed
        finally:
            await runner.cleanup()

    asyncio.run(run())
