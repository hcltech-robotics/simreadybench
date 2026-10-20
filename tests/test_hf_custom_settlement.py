"""HF route declarations settle only against their durable verified reservation."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
import json
import sqlite3
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from simready_bench.engine import custom_model, hf_pricing as hf, request_cost as cost
from simready_bench.engine.inference_gateway import GatewayConfig, GatewayDenied, InferenceGateway, ProjectBudget

MODEL = 'Qwen/Qwen3.8-27B:deepinfra'


def route(model=MODEL):
    price = hf.HF_PRICING[model]
    base, provider = model.rsplit(':', 1)
    return hf.validate_catalog_route({'data': [{
        'id': base, 'architecture': {'input_modalities': ['text', 'image'], 'output_modalities': ['text']},
        'providers': [{'provider': provider, 'status': 'live', 'supports_tools': True,
                       'context_length': price.context_length,
                       'pricing': {'input': price.input_per_million_usd, 'output': price.output_per_million_usd}}],
    }]}, model)


def config(model=MODEL, *, base_url=hf.HF_ORIGIN + '/v1', relay=False):
    price = hf.HF_PRICING.get(model, hf.HF_PRICING[MODEL])
    endpoint = urlsplit(base_url)
    spec = {'schema': 'simready-operator-model-v1', 'model': model, 'base_url': base_url,
            'api_style': 'openai-completions', 'context_window': price.context_length,
            'maximum_output_tokens': None,
            'pricing': {'input_per_million_usd': price.input_per_million_usd,
                        'output_per_million_usd': price.output_per_million_usd},
            'pricing_source': 'https://operator.example/pricing', 'checked': '2026-10-20'}
    binding = {'schema': 'simready-hf-router-relay/v1', 'model': model,
               'upstream_base_url': hf.HF_ORIGIN + '/v1', 'relay_base_url': base_url} if relay else None
    budget = ProjectBudget('simready-bench', 'fixture-authority', True,
                           endpoint.scheme + '://' + endpoint.netloc, model,
                           time.time() + 300, 64, 100, pricing_status='known')
    return GatewayConfig(base_url, model, 'fixture-no-credential',
                         api_style='openai-completions', budget=budget,
                         model_spec=spec, hf_router_relay=binding, native_output_limits=True,
                         max_input_tokens=price.context_length, max_output_tokens=None)


def payload(cfg):
    return {'model': cfg.model, 'messages': [{'role': 'user', 'content': 'offline fixture'}]}


def reserve(tmp_path, monkeypatch, cfg, *, evidence=None, request_id='request'):
    calls = []

    async def fetch(model, *, deadline_at):
        calls.append((model, deadline_at))
        database = tmp_path / 'request-cost.sqlite3'
        if database.exists():
            with sqlite3.connect(database) as db:
                assert db.execute('SELECT COUNT(*) FROM reservations').fetchone()[0] == 0
        return evidence if evidence is not None else route(model)

    monkeypatch.setattr(hf, 'fetch_hf_route', fetch)
    gateway = InferenceGateway(SimpleNamespace(state_dir=tmp_path), cfg)
    reservation = asyncio.run(gateway.reserve_request_cost(payload(cfg), request_id, 'actor', time.time() + 60))
    assert gateway._session is None
    return reservation, calls


def observer(reservation, cfg, *, request_id='request'):
    return {**reservation, 'gateway_request_id': request_id,
            'model': cfg.model, 'provider_reported_models': [cfg.model.rsplit(':', 1)[0]],
            'api_path': '/v1/chat/completions', 'status': 'completed', 'upstream_status': 200,
            'response_sha256': 'f' * 64,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}}


@pytest.mark.parametrize('model', [MODEL, 'google/gemma-4-31B-it:novita'])
@pytest.mark.parametrize('relay', [False, True])
def test_verified_hf_custom_declaration_is_durable_and_settles_without_inference(tmp_path, monkeypatch, model, relay):
    cfg = config(model, base_url='http://127.0.0.1:54321/v1' if relay else hf.HF_ORIGIN + '/v1', relay=relay)
    reservation, calls = reserve(tmp_path, monkeypatch, cfg)
    assert len(calls) == 1 and calls[0][0] == model
    assert reservation['route_metadata_sha256'] == route(model).metadata_sha256
    assert reservation['route_identity_status'] == 'hf_router_catalogue_verified'
    assert reservation['declaration_status'] == 'matched_hf_router_catalogue'
    assert reservation['pricing_source'] == hf.HF_CATALOG_URL
    assert reservation['pricing_checked'] == hf.HF_PRICING_CHECKED
    ledger = cost.CostLedger(tmp_path)
    with sqlite3.connect(ledger.path) as db:
        original_raw = db.execute('SELECT record FROM reservations WHERE request_id=?', ('request',)).fetchone()[0]
    original = json.loads(original_raw)
    assert original['route_metadata_sha256'] == reservation['route_metadata_sha256']
    assert original['hf_router_route']['upstream_base_url'] == hf.HF_ORIGIN + '/v1'
    assert original['hf_router_route']['model'] == model
    evidence = cost.make_usage_settlement_evidence(observer(reservation, cfg))
    settled = ledger.settle('request', evidence=evidence, cost_payload_sha256=cost.sha(payload(cfg)))
    price = hf.HF_PRICING[model]
    expected = int((100 * Decimal(price.input_per_million_usd) + 10 * Decimal(price.output_per_million_usd)).to_integral_value(rounding=ROUND_CEILING))
    assert settled['settled_upper_micro_usd'] == expected
    assert ledger.settle('request', evidence=evidence)['idempotent'] is True
    with sqlite3.connect(ledger.path) as db:
        assert db.execute('SELECT record FROM reservations WHERE request_id=?', ('request',)).fetchone()[0] == original_raw


def test_settlement_rejects_different_route_than_original_durable_reservation(tmp_path, monkeypatch):
    cfg = config()
    reservation, _ = reserve(tmp_path, monkeypatch, cfg)
    changed = observer(reservation, cfg)
    changed['route_metadata_sha256'] = 'a' * 64
    ledger = cost.CostLedger(tmp_path)
    with pytest.raises(cost.CostDenied, match='cost_settlement_original_route_mismatch'):
        ledger.settle('request', evidence=cost.make_usage_settlement_evidence(changed))
    with sqlite3.connect(ledger.path) as db:
        assert db.execute('SELECT COUNT(*) FROM cost_settlements').fetchone()[0] == 0


@pytest.mark.parametrize('change', ['model', 'price', 'source', 'age', 'hash'])
def test_wrong_or_stale_public_route_is_rejected_before_reservation(tmp_path, monkeypatch, change):
    now = time.time()
    monkeypatch.setattr(time, 'time', lambda: now)
    verified = route()
    if change == 'model':
        verified = route('google/gemma-4-31B-it:novita')
    elif change == 'price':
        verified = replace(verified, pricing=replace(verified.pricing, input_per_million_usd='999'))
    elif change == 'source':
        verified = replace(verified, source='https://untrusted.example/models')
    elif change == 'age':
        verified = replace(verified, verified_at=now - 31)
    else:
        verified = replace(verified, metadata_sha256='not-a-sha256')
    with pytest.raises(GatewayDenied, match='hf_model_route_evidence_required'):
        reserve(tmp_path, monkeypatch, config(), evidence=verified)
    with sqlite3.connect(tmp_path / 'request-cost.sqlite3') as db:
        assert db.execute('SELECT COUNT(*) FROM reservations').fetchone()[0] == 0


@pytest.mark.parametrize('change', ['context', 'input_rate', 'output_rate', 'native_output_capacity'])
def test_custom_capacity_and_prices_must_match_independently_verified_route(tmp_path, monkeypatch, change):
    cfg = config()
    spec = deepcopy(cfg.model_spec)
    if change == 'context':
        spec['context_window'] -= 1
    elif change == 'input_rate':
        spec['pricing']['input_per_million_usd'] = '0.1'
    elif change == 'output_rate':
        spec['pricing']['output_per_million_usd'] = '999'
    else:
        spec['maximum_output_tokens'] = 4096
    cfg = replace(cfg, model_spec=spec, max_input_tokens=spec['context_window'])
    with pytest.raises(GatewayDenied, match='hf_model_declaration_route_mismatch'):
        reserve(tmp_path, monkeypatch, cfg)


def test_unknown_hf_router_custom_model_fails_without_forged_evidence(tmp_path):
    cfg = config('example/unknown:provider')
    gateway = InferenceGateway(SimpleNamespace(state_dir=tmp_path), cfg)
    with pytest.raises(GatewayDenied, match='hf_cost_model_not_authorized'):
        asyncio.run(gateway.reserve_request_cost(payload(cfg), 'unknown', 'actor', time.time() + 60))


def test_arbitrary_custom_endpoint_remains_operator_declared_even_for_catalogue_model(tmp_path, monkeypatch):
    cfg = config(base_url='https://models.example.org/v1')
    reservation, calls = reserve(tmp_path, monkeypatch, cfg)
    assert calls == []
    assert reservation['declaration_status'] == 'operator_declared_not_provider_verified'
    assert 'route_metadata_sha256' not in reservation
    forged = observer(reservation, cfg)
    forged.update(route_metadata_sha256=route().metadata_sha256,
                  pricing_source=hf.HF_CATALOG_URL, pricing_checked=hf.HF_PRICING_CHECKED)
    with pytest.raises(cost.CostDenied, match='cost_settlement_original_route_mismatch'):
        cost.CostLedger(tmp_path).settle('request', evidence=cost.make_usage_settlement_evidence(forged))


def test_bare_loopback_catalogue_model_is_not_inferred_to_be_hf_router(tmp_path, monkeypatch):
    cfg = config(base_url='http://127.0.0.1:54321/v1')
    reservation, calls = reserve(tmp_path, monkeypatch, cfg)
    assert calls == []
    assert reservation['declaration_status'] == 'operator_declared_not_provider_verified'
    with pytest.raises(cost.CostDenied, match='hf_model_route_binding_required'):
        custom_model.reserve_custom_cost(cfg, payload(cfg), cost.CostLedger(tmp_path), 'forged', 'actor', route=route())


@pytest.mark.parametrize('change', ['model', 'upstream', 'relay', 'extra'])
def test_private_relay_binding_rejects_wrong_model_endpoint_or_shape(change):
    cfg = config(base_url='http://127.0.0.1:54321/v1', relay=True)
    binding = deepcopy(cfg.hf_router_relay)
    if change == 'model':
        binding['model'] = 'other/model:provider'
    elif change == 'upstream':
        binding['upstream_base_url'] = 'https://models.example.org/v1'
    elif change == 'relay':
        binding['relay_base_url'] = 'http://127.0.0.1:54322/v1'
    else:
        binding['forged'] = True
    with pytest.raises(ValueError, match='HF Router relay'):
        replace(cfg, hf_router_relay=binding)


def test_native_hf_catalogue_reservation_path_remains_unchanged(tmp_path, monkeypatch):
    cfg = replace(config(), model_spec=None)
    reservation, calls = reserve(tmp_path, monkeypatch, cfg)
    assert len(calls) == 1
    assert reservation['reservation_basis'] == 'hf-provider-context-capacity-output-upper-bound'
    assert reservation['route_metadata_sha256'] == route().metadata_sha256
    assert 'model_spec_sha256' not in reservation
    assert cost.CostLedger(tmp_path).settle('request', evidence=cost.make_usage_settlement_evidence(observer(reservation, cfg)))['settled_upper_micro_usd'] == 45


def test_daytona_private_route_survives_envelope_policy_and_gateway(tmp_path, monkeypatch):
    from simready_bench.benchmark import _new_provider
    from simready_bench.config import validate_policy
    from simready_bench.daytona import remote_provider
    from simready_bench.local import _gateway

    parent = _new_provider('huggingface', MODEL, api_key_env=None,
                           maximum_spend_usd=100, conditions=1, authoring_turns=64)
    original = deepcopy(parent)
    remote = remote_provider(parent, port=54321, allocation_id='offline',
                             requests=64, maximum_spend_usd=100)
    assert parent == original
    envelope = json.loads(json.dumps({'provider': remote}))
    roots = {}
    for name in ('assets', 'input_views', 'skills', 'private_manifests'):
        directory = tmp_path / name
        directory.mkdir()
        roots[name] = str(directory)
    operator = validate_policy({'roots': roots,
        'providers': {'selected': envelope['provider']},
        'limits': {'maximum_spend_usd': 100, 'maximum_matrix_runs': 1,
                   'inference_requests': 64, 'max_input_tokens': None, 'max_output_tokens': None}},
        base_dir=tmp_path)
    monkeypatch.setenv('SIMREADY_RELAY_KEY', 'offline-fixture-marker')
    condition = {'condition_id': 'offline-condition', 'authoring_protocol': 'authoring-64-native-output-v1',
                 'model': {'model_id': MODEL, 'provider_ref': 'selected', 'adapter': 'openai-completions'},
                 'harness': {'adapter': 'harbor'},
                 'settings': {'budgets': {'maximum_spend_usd': 100, 'inference_requests': 64,
                                         'agent_seconds': 1800, 'max_input_tokens': 262144, 'max_output_tokens': None}}}
    cfg = _gateway(condition, {'operator_policy': operator})
    assert cfg.hf_router_relay == remote['hf_router_relay']
    assert cfg.upstream_base_url == 'http://127.0.0.1:54321/v1'
    reservation, calls = reserve(tmp_path, monkeypatch, cfg)
    assert calls[0][0] == MODEL
    settled = cost.CostLedger(tmp_path).settle('request', evidence=cost.make_usage_settlement_evidence(observer(reservation, cfg)))
    assert settled['settled_upper_micro_usd'] == 45
