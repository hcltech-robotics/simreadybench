"""Pinned multimodal Nemotron routing and native thinking controls."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from simready_bench.engine.hf_pricing import validate_catalog_route
from simready_bench.engine.inference_gateway import GatewayDenied, InferenceGateway
from simready_bench.engine.request_cost import CostDenied, native_output_facts
from test_hf_cost import config

MODEL = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4:baseten'


def catalog():
    return {'data': [{
        'id': MODEL.rsplit(':', 1)[0],
        'architecture': {'input_modalities': ['text', 'image'], 'output_modalities': ['text']},
        'providers': [{'provider': 'baseten', 'status': 'live', 'supports_tools': True,
                       'context_length': 1048576, 'pricing': {'input': 0.20, 'output': 0.50}}],
    }]}


def test_nemotron_route_pins_the_requested_checkpoint_and_provider_rates():
    route = validate_catalog_route(catalog(), MODEL)
    assert route.pricing.input_per_million_usd == '0.20'
    assert route.pricing.output_per_million_usd == '0.50'
    assert native_output_facts(MODEL)['context_window'] == 1048576
    changed = deepcopy(catalog())
    changed['data'][0]['providers'][0]['pricing']['input'] = 0.21
    with pytest.raises(CostDenied, match='hf_pricing_metadata_changed'):
        validate_catalog_route(changed, MODEL)


def gateway():
    result = object.__new__(InferenceGateway)
    cfg = config()
    result.config = replace(cfg, model=MODEL, budget=replace(cfg.budget, model=MODEL),
                            max_output_tokens=4096, native_output_cap=True)
    return result


@pytest.mark.parametrize('thinking', [
    {'enable_thinking': False},
    {'enable_thinking': False, 'preserve_thinking': True},
    {'enable_thinking': True, 'preserve_thinking': False},
])
def test_native_chat_template_thinking_control_survives_gateway(thinking):
    body = {'model': MODEL, 'messages': [], 'max_tokens': 4096,
            'chat_template_kwargs': thinking}
    assert gateway()._validate_payload(json.dumps(body).encode()) == body


@pytest.mark.parametrize('thinking', [None, True, {}, {'enable_thinking': 0},
    {'enable_thinking': False, 'preserve_thinking': 'true'},
    {'enable_thinking': False, 'endpoint': 'https://elsewhere.example'},
    {'enable_thinking': False, 'system_prompt': 'override'},
])
def test_chat_template_rejects_non_boolean_controls_and_template_injection(thinking):
    body = {'model': MODEL, 'messages': [], 'max_tokens': 4096,
            'chat_template_kwargs': thinking}
    with pytest.raises(GatewayDenied, match='invalid_thinking_setting'):
        gateway()._validate_payload(json.dumps(body).encode())
