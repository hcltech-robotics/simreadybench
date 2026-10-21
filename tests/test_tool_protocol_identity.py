import pytest

from simready_bench.engine.native_output import authoring_protocol, authoring_turns, native_output
from simready_bench.config import condition_identity


def test_bounded_tools_have_a_distinct_scientific_identity():
    legacy = authoring_protocol(64, 8192)
    bounded = authoring_protocol(64, 8192, tool_protocol='bounded-v1')
    assert legacy == 'authoring-64-bounded-output-8192-v1'
    assert bounded == 'authoring-64-bounded-output-8192-v1-tools-bounded-v1'
    assert authoring_turns(bounded) == 64
    labels = {'asset': 'a', 'model': 'm', 'harness': 'pi', 'skill_set': 'none'}
    assert condition_identity(labels, repetition=0, master_seed=1, authoring_protocol=legacy) != condition_identity(labels, repetition=0, master_seed=1, authoring_protocol=bounded)


def test_native_output_cap_detection_retains_bounded_tool_suffix():
    protocol = authoring_protocol(64, tool_protocol='bounded-v1')
    assert native_output({'authoring_protocol': protocol, 'settings': {'generation': {'max_output_tokens': None}}})
    with pytest.raises(ValueError, match='tool'):
        authoring_protocol(64, tool_protocol='unknown')
