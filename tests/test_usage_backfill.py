from copy import deepcopy
from decimal import Decimal

import pytest

from simready_bench.usage_backfill import digest, reconstruct_row


def tariff(**changes):
    return {
        "model": "gpt-6.1-sol",
        "route_verified": True,
        "service_tier_verified": True,
        "pricing_date": "2026-10-20",
        "source": "https://openai.com/api/pricing/",
        "usd_per_million_tokens": {
            "input": "2",
            "cached_input": "0.1",
            "cache_write": "2.5",
            "output": "10",
        },
        "long_context_threshold": 272_000,
        "long_context_input_multiplier": "2",
        "long_context_output_multiplier": "1.5",
        **changes,
    }


def request(identity, prompt=100, completion=20, cached=50, reasoning=10):
    return {
        "gateway_request_id": identity,
        "model": "gpt-6.1-sol",
        "status": "completed",
        "upstream_status": 200,
        "elapsed_seconds": 1.25,
        "usage": {
            "input_tokens": prompt,
            "output_tokens": completion,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


def trajectory(records, *, writes=None):
    writes = writes if writes is not None else [0] * len(records)
    steps = []
    for record, write in zip(records, writes, strict=True):
        usage = record["usage"]
        steps.append({"metrics": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "cached_tokens": usage["input_tokens_details"]["cached_tokens"],
            "extra": {
                "cache_write_tokens": write,
                "reasoning_tokens": usage["output_tokens_details"]["reasoning_tokens"],
            },
        }})
    return {
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": sum(step["metrics"]["prompt_tokens"] for step in steps),
            "total_completion_tokens": sum(step["metrics"]["completion_tokens"] for step in steps),
            "total_cached_tokens": sum(step["metrics"]["cached_tokens"] for step in steps),
            "extra": {
                "cache_write_tokens": sum(writes),
                "reasoning_tokens": sum(step["metrics"]["extra"]["reasoning_tokens"] for step in steps),
            },
        },
    }


def row(records=None, **changes):
    records = [request("request-1")] if records is None else records
    return {
        "condition_id": "canonical-condition",
        "trial_id": "canonical-trial",
        "seed": 42,
        "repetition": 2,
        "reasoning_effort": "medium",
        "simready_reward": 75,
        "candidate": {"manifest_sha256": "a" * 64},
        "inference_calls_used": len(records),
        "model_requests": records,
        **changes,
    }


def recover(source, trace=None, pricing=None):
    return reconstruct_row(
        source,
        run_id="canonical-run",
        trajectory=trace,
        trajectory_sha256=digest(trace) if trace is not None else None,
        tariff=pricing,
    )


def test_provider_usage_prices_ordinary_cache_and_output_without_double_counting():
    def split(total):
        quotient, remainder = divmod(total, 20)
        return [quotient + (index < remainder) for index in range(20)]

    inputs, outputs, cached, writes = map(split, (361_727, 10_811, 330_337, 31_330))
    records = [
        request(str(index), prompt, output, cache, reasoning=250)
        for index, (prompt, output, cache) in enumerate(zip(inputs, outputs, cached, strict=True))
    ]
    source = row(records)
    original = deepcopy(source)
    result = recover(source, trajectory(records, writes=writes), tariff())
    ordinary = 361_727 - 330_337 - 31_330
    assert ordinary == 60
    expected = (Decimal(ordinary) * 2 + Decimal(330_337) / 10 + Decimal(31_330) * Decimal("2.5") + Decimal(10_811) * 10) / 1_000_000
    assert expected == Decimal("0.2195887")
    assert result["inference_cost_usd"] == pytest.approx(float(expected))
    assert result["input_tokens"] == 361_727
    assert result["output_tokens"] == 10_811
    assert result["total_tokens"] == 372_538
    assert result["usage_reconstruction"]["recorded_calls"] == 20
    assert result["usage_reconstruction"]["pricing_derived_cost_complete"] is True
    assert result["inference_cost_kind"] == "pricing-derived inference cost"
    assert "compute_cost_usd" not in result
    assert "total_cost_usd" not in result
    assert source == original
    for field in ("condition_id", "trial_id", "seed", "repetition", "reasoning_effort", "simready_reward", "candidate"):
        assert result[field] == source[field]


def test_context_tiers_are_applied_per_request_at_the_strict_threshold():
    records = [
        request("at-threshold", 272_000, 10, 0, 0),
        request("above-threshold", 272_001, 10, 0, 0),
    ]
    result = recover(row(records), trajectory(records), tariff())
    expected = (Decimal(272_000) * 2 + 10 * 10 + Decimal(272_001) * 2 * 2 + 10 * 10 * Decimal("1.5")) / 1_000_000
    assert result["inference_cost_usd"] == pytest.approx(float(expected))
    prices = result["usage_reconstruction"]["requests"]
    assert prices[0]["pricing_derived_inference_cost_usd"] == pytest.approx(0.5441)
    assert prices[1]["pricing_derived_inference_cost_usd"] == pytest.approx(1.088154)


def test_identical_request_ids_deduplicate_and_keep_the_admitted_call_count():
    record = request("one")
    source = row([record, deepcopy(record)], inference_calls_used=1)
    result = recover(source, trajectory([record]), tariff())
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 20
    provenance = result["usage_reconstruction"]
    assert provenance["unique_requests"] == 1
    assert provenance["duplicate_records"] == 1
    assert provenance["provider_usage_complete"] is True
    assert len(provenance["requests"]) == 1


def test_conflicting_duplicate_request_ids_are_rejected():
    original = request("one")
    changed = deepcopy(original)
    changed["usage"]["input_tokens"] = 101
    with pytest.raises(ValueError, match="Conflicting records"):
        recover(row([original, changed], inference_calls_used=1), pricing=tariff())


def test_request_without_a_canonical_identifier_is_rejected():
    record = request("one")
    del record["gateway_request_id"]
    with pytest.raises(ValueError, match="canonical identifier"):
        recover(row([record]))


def test_missing_provider_usage_or_missing_call_keeps_recovered_totals_unknown():
    record = request("one")
    missing = request("two")
    del missing["usage"]
    result = recover(row([record, missing]), pricing=tariff())
    assert result.get("input_tokens") is None
    assert result.get("output_tokens") is None
    assert result.get("total_tokens") is None
    assert result.get("inference_cost_usd") is None
    assert result["usage_reconstruction"]["provider_usage_complete"] is False
    incomplete = recover(row([record], inference_calls_used=2), trajectory([record]), tariff())
    assert incomplete.get("input_tokens") is None
    assert incomplete.get("inference_cost_usd") is None
    assert incomplete["usage_reconstruction"]["pricing_derived_cost_complete"] is False


@pytest.mark.parametrize("missing", ["route_verified", "service_tier_verified"])
def test_unverified_price_route_or_service_keeps_cost_unknown(missing):
    record = request("one")
    pricing = tariff(**{missing: False})
    result = recover(row([record]), trajectory([record]), pricing)
    assert result["input_tokens"] == 100
    assert result.get("inference_cost_usd") is None
    assert result["usage_reconstruction"]["pricing_derived_cost_complete"] is False


def test_missing_trajectory_cache_writes_does_not_assume_zero_cost():
    record = request("one")
    result = recover(row([record]), pricing=tariff())
    assert result["input_tokens"] == 100
    assert result.get("inference_cost_usd") is None
    assert result["usage_reconstruction"]["requests"][0]["cache_write_tokens"] is None


def test_existing_top_level_and_nested_usage_are_preserved_and_conflicts_recorded():
    record = request("one")
    source = row(
        [record], input_tokens=900, output_tokens=None, total_tokens=950,
        inference_cost_usd=0,
        statistics={"input_tokens": 800, "output_tokens": 50, "total_tokens": 850, "inference_cost_usd": 7},
    )
    result = recover(source, trajectory([record]), tariff())
    assert result["input_tokens"] == 900
    assert result["output_tokens"] == 50
    assert result["total_tokens"] == 950
    assert result["inference_cost_usd"] == 0
    assert result["statistics"]["input_tokens"] == 800
    assert result["statistics"]["output_tokens"] == 50
    assert result["statistics"]["total_tokens"] == 850
    assert result["statistics"]["inference_cost_usd"] == 7
    assert {"input_tokens", "output_tokens", "total_tokens", "inference_cost_usd"} <= set(result["usage_reconstruction"]["preserved_usage_conflicts"])


def test_backfill_is_idempotent_and_keeps_recorded_provenance_hashes():
    record = request("one")
    source = row([record])
    trace = trajectory([record], writes=[25])
    first = recover(source, trace, tariff())
    second = recover(first, trace, tariff())
    assert second == first
    provenance = first["usage_reconstruction"]
    assert provenance["gateway_records_sha256"] == digest(source["model_requests"])
    assert provenance["trajectory_sha256"] == digest(trace)
    assert provenance["canonical_run_id"] == "canonical-run"
    assert provenance["condition_id"] == source["condition_id"]


def test_nested_usage_conflicts_are_reported_when_the_top_level_matches_provider():
    record = request("one")
    source = row([record], input_tokens=100, statistics={"input_tokens": 800})
    result = recover(source, trajectory([record]), tariff())
    assert result["input_tokens"] == 100
    assert result["statistics"]["input_tokens"] == 800
    assert "input_tokens" in result["usage_reconstruction"]["preserved_usage_conflicts"]


def test_newly_derived_cost_keeps_its_kind_when_recorded_tokens_conflict():
    record = request("one")
    result = recover(row([record], input_tokens=900), trajectory([record]), tariff())
    assert result["inference_cost_usd"] is not None
    assert result["inference_cost_kind"] == "pricing-derived inference cost"


def test_provider_trajectory_mismatch_and_inconsistent_final_totals_are_rejected():
    record = request("one")
    trace = trajectory([record])
    trace["steps"][0]["metrics"]["prompt_tokens"] = 101
    trace["final_metrics"]["total_prompt_tokens"] = 101
    with pytest.raises(ValueError, match="Provider and trajectory usage differ"):
        recover(row([record]), trace, tariff())
    trace = trajectory([record])
    trace["final_metrics"]["total_completion_tokens"] = 21
    with pytest.raises(ValueError, match="Trajectory final usage differs"):
        recover(row([record]), trace, tariff())


def test_cache_details_exceeding_input_tokens_are_rejected():
    record = request("one", prompt=100, cached=80)
    with pytest.raises(ValueError, match="Cache details exceed input"):
        recover(row([record]), trajectory([record], writes=[30]), tariff())


def test_retry_error_counts_include_each_unique_admitted_request():
    failed = request("failed")
    failed.update(status="failed", upstream_status=429, error_code="provider_error")
    del failed["usage"]
    success = request("success")
    result = recover(row([failed, success]))
    assert result["model_request_errors"] == 1
    assert result["model_request_seconds"] == 2.5
    assert result.get("input_tokens") is None
    assert result.get("inference_cost_usd") is None


def test_null_statistics_are_recoverable_and_gateway_aliases_are_retained():
    record = request("one")
    source = row([], statistics=None, inference_calls_used=1, gateway_records=[record])
    result = recover(source, trajectory([record]), tariff())
    assert result["input_tokens"] == result["statistics"]["input_tokens"] == 100
    assert result["total_tokens"] == 120


@pytest.mark.parametrize(
    "change, message",
    [
        ({"prompt": 100, "cached": 101}, "Cached tokens exceed input"),
        ({"completion": 20, "reasoning": 21}, "Reasoning tokens exceed output"),
    ],
)
def test_usage_conservation_is_checked_even_without_a_tariff(change, message):
    record = request("one", **change)
    with pytest.raises(ValueError, match=message):
        recover(row([record]))


def test_inconsistent_provider_total_tokens_are_rejected():
    record = request("one")
    record["usage"]["total_tokens"] = 121
    with pytest.raises(ValueError, match="Provider total tokens differ"):
        recover(row([record]))


def test_error_placeholders_are_unavailable_usage_and_do_not_create_zero_totals():
    success = request("success")
    failed = request("failed")
    failed.update(status="failed", upstream_status=None, error_code="stream_error")
    del failed["usage"]
    trace = trajectory([success])
    trace["steps"].append({
        "metrics": {
            "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
            "extra": {"cache_write_tokens": 0, "reasoning_tokens": 0},
        },
        "extra": {"stop_reason": "error"},
    })
    result = recover(row([success, failed]), trace, tariff())
    assert result.get("input_tokens") is None
    assert result.get("inference_cost_usd") is None
    assert result["usage_reconstruction"]["provider_usage_complete"] is False
    assert result["usage_reconstruction"]["trajectory_usage_matched"] is True


def test_request_model_must_match_the_tariff_model():
    record = request("one")
    with pytest.raises(ValueError, match="model differs from the pricing tariff"):
        recover(row([record]), trajectory([record]), tariff(model="different-model"))
