"""Versioned API token estimates, never provider invoices or spending caps."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path

POLICY = "openai-standard-global-30m"
PRICING_FILE = Path(__file__).with_name("pricing.json")


def estimate_cost(snapshot, model, policy):
    unavailable = lambda reason: {"status": "unavailable", "reason": reason, "billed": False}
    prices = json.loads(PRICING_FILE.read_text())
    if policy != POLICY or policy != prices["policy"]:
        return unavailable("unknown_pricing_policy")
    if model not in prices["models"]:
        return unavailable("unknown_model_pricing")
    if snapshot.get("status") != "verified" or not snapshot.get("requests_complete"):
        return unavailable(snapshot.get("pricing_usage_error") or "unverified_per_request_usage")
    requests = snapshot.get("requests") or []
    totals = snapshot.get("totals") or {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "cache_write_input_tokens", "reasoning_output_tokens"):
        if key not in totals:
            continue
        if sum(r["usage"].get(key, 0) for r in requests) != totals.get(key):
            return unavailable("request_totals_do_not_reconcile")
    rates, amount, breakdown, assumptions = prices["models"][model], Decimal(0), [], set()
    for request in requests:
        usage, context = request["usage"], request["context"]
        if context.get("model") != model:
            return unavailable("request_model_mismatch")
        tier = context.get("service_tier")
        if tier is None:
            tier = "default"
            assumptions.add("Standard service tier when not present in native evidence")
        if not isinstance(tier, str) or tier not in rates["tiers"]:
            return unavailable("unknown_service_tier")
        regional = context.get("regional_processing")
        if regional is None:
            regional = False
            assumptions.add("Global processing when not present in native evidence")
        if type(regional) is not bool:
            return unavailable("unknown_regional_processing")
        inp, cached, output = (usage.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens"))
        writes = usage.get("cache_write_input_tokens", 0)
        if any(type(v) is not int or v < 0 for v in (inp, cached, output, writes)) or cached+writes > inp:
            return unavailable("invalid_request_counters")
        if "cache_write_input_tokens" not in usage:
            assumptions.add("Unreported cache-write tokens use the pinned Codex protocol default of zero")
        retention = context.get("prompt_cache_retention")
        if retention is None:
            retention = "30m"
            if writes:
                assumptions.add("30-minute cache retention when writes are reported without retention metadata")
        if not isinstance(retention, str):
            return unavailable("unknown_cache_write_retention")
        if writes and retention not in {"30m", "12h"}:
            return unavailable("unknown_cache_write_retention")
        long = inp > prices["long_context_threshold"]
        input_factor = Decimal(prices["long_context_input_multiplier"]) if long else Decimal(1)
        output_factor = Decimal(prices["long_context_output_multiplier"]) if long else Decimal(1)
        tier_factor = Decimal(prices["tiers"][tier])
        regional_factor = Decimal(prices["regional_multiplier"]) if regional else Decimal(1)
        charge = ((Decimal(inp-cached-writes)*Decimal(rates["input"]) +
                   Decimal(cached)*Decimal(rates["cached"]) +
                   Decimal(writes)*Decimal(rates.get("write_"+retention, rates["write_30m"]))) * input_factor +
                  Decimal(output)*Decimal(rates["output"])*output_factor) * tier_factor * regional_factor / Decimal(1000000)
        amount += charge
        breakdown.append({"usage": usage, "tier": tier, "regional_processing": regional,
                          "cache_retention": retention, "long_context": long,
                          "amount_usd": str(charge), "evidence": request["evidence"]})
    return {"status": "estimated", "billed": False, "amount_usd": float(amount),
            "amount_decimal_usd": str(amount), "currency": "USD", "policy": POLICY,
            "model": model, "pricing_checked_at": prices["checked_at"], "sources": prices["sources"],
            "pricing_sha256": hashlib.sha256(PRICING_FILE.read_bytes()).hexdigest(),
            "request_count": len(requests), "long_context_requests": sum(r["long_context"] for r in breakdown),
            "totals": totals, "assumptions": sorted(assumptions), "requests": breakdown}
