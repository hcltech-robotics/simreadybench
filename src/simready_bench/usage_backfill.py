"""Recover retained provider usage without changing benchmark measurements."""

from collections import Counter
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math


SCHEMA = "simreadybench-usage-reconstruction/v1"
FIELDS = (
    "input_tokens", "output_tokens", "total_tokens", "model_request_seconds",
    "model_request_errors", "inference_cost_usd",
)


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _sum(values):
    return sum(values) if values and all(v is not None for v in values) else None


def _usage(record):
    usage = record.get("usage") or {}
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
    output = usage.get("output_tokens_details", usage.get("completion_tokens_details")) or {}
    result = {
        "input_tokens": _count(usage.get("input_tokens", usage.get("prompt_tokens"))),
        "output_tokens": _count(usage.get("output_tokens", usage.get("completion_tokens"))),
        "cached_tokens": _count(details.get("cached_tokens")),
        "reasoning_tokens": _count(output.get("reasoning_tokens")),
    }
    if all(result[field] is not None for field in ("input_tokens", "output_tokens")):
        total = _count(usage.get("total_tokens"))
        if total is not None and total != result["input_tokens"] + result["output_tokens"]:
            raise ValueError("Provider total tokens differ from input and output")
    return result


def _trajectory_usage(trajectory):
    calls = []
    for step in trajectory.get("steps", []):
        metrics = step.get("metrics") or {}
        if metrics.get("prompt_tokens") is None and metrics.get("completion_tokens") is None:
            continue
        extra = metrics.get("extra") or {}
        # Pi emits zero placeholders after a stream without a terminal usage
        # event. The retained stop reason identifies these as unavailable.
        if (step.get("extra") or {}).get("stop_reason") == "error" and all(
            metrics.get(field) == 0 for field in ("prompt_tokens", "completion_tokens", "cached_tokens")
        ):
            continue
        calls.append({
            "input_tokens": _count(metrics.get("prompt_tokens")),
            "output_tokens": _count(metrics.get("completion_tokens")),
            "cached_tokens": _count(metrics.get("cached_tokens")),
            "cache_write_tokens": _count(extra.get("cache_write_tokens")),
            "reasoning_tokens": _count(extra.get("reasoning_tokens")),
        })
    final = trajectory.get("final_metrics") or {}
    for field, name in (
        ("input_tokens", "total_prompt_tokens"),
        ("output_tokens", "total_completion_tokens"),
        ("cached_tokens", "total_cached_tokens"),
    ):
        actual = _sum([call[field] for call in calls])
        expected = _count(final.get(name))
        if actual is not None and expected is not None and actual != expected:
            raise ValueError("Trajectory final usage differs from its steps: " + field)
    for field in ("cache_write_tokens", "reasoning_tokens"):
        actual = _sum([call[field] for call in calls])
        expected = _count((final.get("extra") or {}).get(field))
        if actual is not None and expected is not None and actual != expected:
            raise ValueError("Trajectory final usage differs from its steps: " + field)
    return calls


def _key(usage):
    return tuple(usage.get(field) for field in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens"))


def _price(usage, tariff):
    _validate_usage(usage)
    if tariff and tariff.get("billing_basis") == "input_output":
        if not tariff.get("route_verified") or not tariff.get("service_tier_verified"):
            return None
        if any(usage.get(field) is None for field in ("input_tokens", "output_tokens")):
            return None
        rates = tariff["usd_per_million_tokens"]
        values = [Decimal(str(rates[field])) for field in ("input", "output")]
        if any(not rate.is_finite() or rate < 0 for rate in values):
            raise ValueError("Token prices must be finite and non-negative")
        return (usage["input_tokens"] * values[0] + usage["output_tokens"] * values[1]) / Decimal(1_000_000)
    required = ("input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens")
    if not tariff or not all(usage.get(field) is not None for field in required):
        return None
    if not tariff.get("route_verified") or not tariff.get("service_tier_verified"):
        return None
    ordinary = usage["input_tokens"] - usage["cached_tokens"] - usage["cache_write_tokens"]
    if ordinary < 0:
        raise ValueError("Cache details exceed input tokens")
    rates = tariff["usd_per_million_tokens"]
    long_context = usage["input_tokens"] > tariff["long_context_threshold"]
    input_multiplier = Decimal(str(tariff["long_context_input_multiplier"] if long_context else 1))
    output_multiplier = Decimal(str(tariff["long_context_output_multiplier"] if long_context else 1))
    cost = input_multiplier * (
        ordinary * Decimal(str(rates["input"]))
        + usage["cached_tokens"] * Decimal(str(rates["cached_input"]))
        + usage["cache_write_tokens"] * Decimal(str(rates["cache_write"]))
    ) + output_multiplier * usage["output_tokens"] * Decimal(str(rates["output"]))
    return cost / Decimal(1_000_000)


def _validate_usage(usage):
    input_tokens, cached, writes = (usage.get(field) for field in ("input_tokens", "cached_tokens", "cache_write_tokens"))
    if input_tokens is not None:
        if cached is not None and cached > input_tokens:
            raise ValueError("Cached tokens exceed input tokens")
        if writes is not None and writes > input_tokens:
            raise ValueError("Cache write tokens exceed input tokens")
        if cached is not None and writes is not None and cached + writes > input_tokens:
            raise ValueError("Cache details exceed input tokens")
    if usage.get("reasoning_tokens") is not None and usage.get("output_tokens") is not None and usage["reasoning_tokens"] > usage["output_tokens"]:
        raise ValueError("Reasoning tokens exceed output tokens")


def reconstruct_row(row, *, run_id, trajectory=None, trajectory_sha256=None, tariff=None, expected_request_count=None):
    """Fill missing usage and retain request, trajectory and tariff provenance.

    Request identifiers deduplicate retained records. A complete cost requires
    one provider usage record per admitted call, a matching trajectory cache
    breakdown and verified route and service pricing. Unknown calls remain
    unknown. Cache and reasoning details are subsets of the reported totals.
    """
    result = deepcopy(row)
    original_records = row.get("model_requests") or row.get("gateway_records") or []
    records = {}
    for record in original_records:
        if tariff and record.get("model") != tariff.get("model"):
            raise ValueError("Provider record model differs from the pricing tariff")
        identity = record.get("gateway_request_id")
        if not identity:
            raise ValueError("A retained provider request has no canonical identifier")
        if identity in records and records[identity] != record:
            raise ValueError("Conflicting records for provider request " + identity)
        records[identity] = record
    requests = list(records.values())
    if expected_request_count is not None and _count(expected_request_count) is None:
        raise ValueError("Expected request count must be a non-negative integer")
    expected = _count(row.get("inference_calls_used")) if expected_request_count is None else expected_request_count
    complete_count = expected is not None and len(requests) == expected
    usages = [_usage(record) for record in requests]
    trajectory_calls = _trajectory_usage(trajectory) if trajectory is not None else []
    observed = [usage for usage in usages if usage["input_tokens"] is not None and usage["output_tokens"] is not None]
    comparison_observed = [_usage({"usage": record.get("usage") or record.get("partial_usage")}) for record in requests]
    comparison_observed = [usage for usage in comparison_observed if usage["input_tokens"] is not None and usage["output_tokens"] is not None]
    trajectory_matched = bool(trajectory_calls) and Counter(map(_key, comparison_observed)) == Counter(map(_key, trajectory_calls))
    if trajectory_calls and not trajectory_matched:
        raise ValueError("Provider and trajectory usage differ for " + row["condition_id"])
    cache_writes = {}
    for usage in trajectory_calls:
        cache_writes.setdefault(_key(usage), []).append(usage["cache_write_tokens"])
    reconstructed_requests = []
    prices = []
    for record, usage in zip(requests, usages, strict=True):
        writes = cache_writes.get(_key(usage))
        usage["cache_write_tokens"] = writes.pop(0) if writes else None
        price = _price(usage, tariff)
        prices.append(price)
        reconstructed_requests.append({
            "gateway_request_id": record["gateway_request_id"],
            **usage,
            "pricing_derived_inference_cost_usd": float(price) if price is not None else None,
        })
    recovered = {
        field: _sum([usage[field] for usage in usages]) if complete_count else None
        for field in ("input_tokens", "output_tokens")
    }
    recovered["total_tokens"] = (
        recovered["input_tokens"] + recovered["output_tokens"]
        if all(recovered[field] is not None for field in ("input_tokens", "output_tokens")) else None
    )
    seconds = [record.get("elapsed_seconds") for record in requests]
    recovered["model_request_seconds"] = sum(seconds) if complete_count and seconds and all(
        type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in seconds
    ) else None
    recovered["model_request_errors"] = sum(
        bool(record.get("error_code")) or record.get("status") != "completed"
        or type(record.get("upstream_status")) is int and record["upstream_status"] >= 400
        for record in requests
    ) if complete_count and requests else None
    cost_complete = complete_count and bool(prices) and all(price is not None for price in prices)
    recovered["inference_cost_usd"] = float(sum(prices, Decimal(0))) if cost_complete else None
    statistics = result.get("statistics")
    if not isinstance(statistics, dict):
        statistics = result["statistics"] = {}
    conflicts = []
    for field in FIELDS:
        prior = row.get(field)
        nested = statistics.get(field)
        value = prior if prior is not None else nested if nested is not None else recovered[field]
        if recovered[field] is not None and any(
            recorded is not None and not math.isclose(recorded, recovered[field], rel_tol=1e-12, abs_tol=1e-9)
            for recorded in (prior, nested)
        ):
            conflicts.append(field)
        if result.get(field) is None and value is not None:
            result[field] = value
        if statistics.get(field) is None and value is not None:
            statistics[field] = value
    result["usage_reconstruction"] = {
        "schema": SCHEMA,
        "canonical_run_id": run_id,
        "condition_id": row["condition_id"],
        "trial_id": row.get("trial_id"),
        "seed": row.get("seed"),
        "gateway_records_sha256": digest(original_records),
        "trajectory_sha256": trajectory_sha256,
        "recorded_calls": expected,
        "productive_calls": row.get("inference_calls_used"),
        "unique_requests": len(requests),
        "duplicate_records": len(original_records) - len(requests),
        "provider_usage_complete": complete_count and len(observed) == len(requests),
        "trajectory_usage_matched": trajectory_matched,
        "pricing_derived_cost_complete": cost_complete,
        "measured_pricing_derived_cost_usd": float(sum((price for price in prices if price is not None), Decimal(0))) if any(price is not None for price in prices) else None,
        "priced_requests": sum(price is not None for price in prices),
        "preserved_usage_conflicts": conflicts,
        "tariff": deepcopy(tariff),
        "requests": reconstructed_requests,
    }
    if cost_complete and "inference_cost_usd" not in conflicts and result.get("inference_cost_kind") is None:
        result["inference_cost_kind"] = "pricing-derived inference cost"
    return result
