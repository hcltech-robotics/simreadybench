import pytest

from simready_bench.daytona import _validate_relay_ledger


def ledger():
    return {
        "model": "model", "generation_requests": 64, "input_counter_requests": 0,
        "maximum_requests": 64, "maximum_condition_requests": 64,
        "expires_at": 123, "events": [], "condition_generation_requests": {},
        "condition_input_counter_requests": {}, "condition_limits": {"condition": 64},
    }


def test_collection_accepts_more_attempts_than_productive_calls():
    value = ledger()
    value.update(accounting_version=2, model_calls=64,
                 condition_model_calls={"condition": 64},
                 maximum_transport_requests=256, generation_requests=95)
    _validate_relay_ledger(value, {"planned_requests": 64})


def test_collection_retains_legacy_attempt_bound():
    value = ledger()
    _validate_relay_ledger(value, {"planned_requests": 64})
    value["generation_requests"] = 65
    with pytest.raises(ValueError):
        _validate_relay_ledger(value, {"planned_requests": 64})


@pytest.mark.parametrize("changed", [
    {"maximum_transport_requests": 257}, {"model_calls": 65},
    {"condition_model_calls": {"condition": 65}}, {"generation_requests": 257},
    {"condition_model_calls": {"unknown": 1}}, {"accounting_version": 3},
])
def test_collection_rejects_expanded_authority(changed):
    value = ledger()
    value.update(accounting_version=2, model_calls=64,
                 condition_model_calls={"condition": 64}, maximum_transport_requests=256)
    value.update(changed)
    with pytest.raises(ValueError):
        _validate_relay_ledger(value, {"planned_requests": 64})
