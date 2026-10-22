"""Continuation is an explicit protocol without changing per-call output caps."""
import pytest
from simready_bench.engine.native_output import authoring_protocol,authoring_turns
from simready_bench.publishing import _viewer_protocol

@pytest.mark.parametrize('tools',['legacy','bounded-v1'])
@pytest.mark.parametrize('cap',[None,4096])
def test_length_continuation_protocol_is_round_trippable_and_distinct(tools,cap):
    original=authoring_protocol(64,cap,tool_protocol=tools)
    continued=authoring_protocol(64,cap,tool_protocol=tools,continue_on_length=True)
    assert continued==original+'-continue-length-v1'
    assert authoring_turns(original)==authoring_turns(continued)==64
    viewer=_viewer_protocol({'authoring_protocol':continued},{},'fixture')
    assert viewer['max_output_tokens']==(cap if cap is not None else '')
    assert viewer['output_limit']==('bounded' if cap else 'provider-native')

def test_length_continuation_does_not_accept_truthy_non_booleans():
    with pytest.raises(ValueError,match='explicit boolean'):
        authoring_protocol(64,4096,continue_on_length='yes')
