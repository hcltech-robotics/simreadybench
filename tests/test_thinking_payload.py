"""Explicit native thinking settings survive the trusted chat gateway."""
from dataclasses import replace
import json
import pytest
from simready_bench.engine.inference_gateway import InferenceGateway,GatewayDenied
from test_hf_cost import config

@pytest.mark.parametrize('relay',[False,True])
@pytest.mark.parametrize('thinking',[{'type':'disabled'},{'type':'enabled','clear_thinking':False}])
def test_native_thinking_setting_passes_without_changing_request_or_cap(relay,thinking):
    gateway=object.__new__(InferenceGateway)
    cfg=config()
    cfg=replace(cfg,model='zai-org/GLM-5.3-Flash:baseten',budget=replace(cfg.budget,model='zai-org/GLM-5.3-Flash:baseten'),max_output_tokens=4096,native_output_cap=True)
    if relay:
        object.__setattr__(cfg,'upstream_base_url','http://127.0.0.1:9999/v1')
        object.__setattr__(cfg,'hf_router_relay',{'model':cfg.model})
    gateway.config=cfg
    body={'model':cfg.model,'messages':[{'role':'user','content':'asset'}],
          'max_tokens':4096,'thinking':thinking}
    assert gateway._validate_payload(json.dumps(body).encode())==body

@pytest.mark.parametrize('thinking',[None,True,'disabled',{}, {'type':'maybe'},
    {'type':'disabled','endpoint':'https://elsewhere.example'},
    {'type':'enabled','clear_thinking':1}])
def test_thinking_controls_have_a_strict_transport_safe_shape(thinking):
    gateway=object.__new__(InferenceGateway)
    gateway.config=replace(config(),max_output_tokens=4096,native_output_cap=True)
    body={'model':gateway.config.model,'messages':[], 'max_tokens':4096,'thinking':thinking}
    with pytest.raises(GatewayDenied,match='invalid_thinking_setting'):
        gateway._validate_payload(json.dumps(body).encode())
