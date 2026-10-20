"""Settle completed model requests from recorded usage."""

import pytest
from simready_bench.engine import request_cost as c


def record(model="gpt-6-astra"):
    hf = ":" in model
    upper = 62669 if hf else 217300
    return {
        "gateway_request_id": "r",
        "model": model,
        "provider_reported_models": [model.split(":")[0]],
        "api_path": "/v1/chat/completions" if hf else "/v1/responses",
        "status": "completed",
        "upstream_status": 200,
        "request_sha256": "b" * 64,
        "response_sha256": "c" * 64,
        "pricing_checked": "2026-10-18",
        "pricing_source": "https://router.huggingface.co/v1/models"
        if hf
        else c.PRICING_SOURCES[model],
        "route_metadata_sha256": "d" * 64 if hf else None,
        "max_output_tokens": 4096,
        "reserved_upper_cost_micro_usd": upper,
        "usage": {"prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104}
        if hf
        else {"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010},
        "shared_asset_reservation": {
            "request_id": "r",
            "reserved_upper_cost_micro_usd": upper,
        },
    }


def verify(r, **changes):
    args = {
        "request_id": "r",
        "original_upper_micro_usd": r["reserved_upper_cost_micro_usd"],
    }
    args.update(changes)
    return c.verify_usage_settlement(c.make_usage_settlement_evidence(r), **args)


def test_usage_upper_bound_releases_unused_output_reserve_without_invoice_claim():
    p = verify(record())
    assert p["settled_upper_micro_usd"] == 13000
    assert p["cost_basis"] == "provider_usage_priced_upper_bound"
    assert p["invoice_verified"] is False
    assert p["proof_sha256"] == c.sha(p["proof"])
    assert p["proof"]["original_upper_micro_usd"] == 217300
    assert p["proof"]["unused_upper_micro_usd"] == 204300


def test_hf_actual_prompt_and_completion_usage_uses_pinned_route_prices():
    p = verify(record("Qwen/Qwen3.8-27B:deepinfra"))
    assert p["settled_upper_micro_usd"] == 30
    assert p["proof"]["pricing_source"] == "https://router.huggingface.co/v1/models"


@pytest.mark.parametrize(
    "change",
    [
        {"status": "cancelled"},
        {"upstream_status": 500},
        {"usage": {}},
        {"provider_reported_models": []},
        {"provider_reported_models": ["XiaomiMiMo/MiMo-V2.6-Flash"]},
        {"pricing_checked": "2026-10-17"},
        {"pricing_source": "https://untrusted.example"},
        {"request_sha256": "wrong"},
        {"response_sha256": "wrong"},
    ],
)
def test_unknown_incomplete_or_unbound_observations_cannot_settle(change):
    r = record()
    r.update(change)
    with pytest.raises(c.CostDenied):
        verify(r)


@pytest.mark.parametrize(
    "change", [{"request_id": "other"}, {"original_upper_micro_usd": 217301}]
)
def test_settlement_binds_original_immutable_attempt_and_approval(change):
    with pytest.raises(c.CostDenied):
        verify(record(), **change)


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True, "output_tokens": 1, "total_tokens": 2},
        {"input_tokens": -1, "output_tokens": 1, "total_tokens": 0},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 3},
        {"input_tokens": 1.0, "output_tokens": 1, "total_tokens": 2},
    ],
)
def test_usage_must_be_exact_finite_nonnegative_integer_counts(usage):
    r = record()
    r["usage"] = usage
    with pytest.raises(c.CostDenied):
        verify(r)


def test_cache_detail_absence_keeps_highest_input_rate_and_never_claims_invoice():
    r = record()
    r["usage"]["input_tokens_details"] = {"cached_tokens": 1000}
    assert verify(r)["settled_upper_micro_usd"] == 13000
    assert verify(r)["invoice_verified"] is False


def test_observer_declared_actual_usd_cannot_override_verified_usage_calculation():
    r = record()
    r["actual_cost_usd"] = 0
    r["usage"]["cost"] = 0
    assert verify(r)["settled_upper_micro_usd"] == 13000


def test_usage_cost_above_original_hold_does_not_hide_overshoot():
    r = record()
    r["reserved_upper_cost_micro_usd"] = 1
    r["shared_asset_reservation"]["reserved_upper_cost_micro_usd"] = 1
    with pytest.raises(c.CostDenied, match="usage_cost_exceeds_original_hold"):
        verify(r)


def test_flash_mapping_is_not_case_normalization():
    r = record("Qwen/Qwen3.8-27B:deepinfra")
    r["model"] = "zai-org/GLM-4.6V-Flash:novita"
    r["provider_reported_models"] = ["zai-org/glm-4.6v"]
    with pytest.raises(c.CostDenied):
        verify(r)


def test_dictionary_cannot_bypass_typed_host_observer_boundary():
    with pytest.raises(c.CostDenied):
        c.verify_usage_settlement(
            record(), request_id="r", original_upper_micro_usd=217300
        )
