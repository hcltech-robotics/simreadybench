"""Final native Pi receipts for ACP; per-call usage, never context occupancy."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from .agent import NvidiaPi


def baseten_reference(receipt, model_id):
    """A reference comparison, never an NVIDIA invoice or ACP billed cost."""
    if model_id != "nvidia/nemotron-3.5-super-vl-120b-a12b" or receipt.get("status") != "verified":
        return None
    usage = receipt.get("usage") or {}
    prompt, output = usage.get("inputTokens"), usage.get("outputTokens")
    cached = usage.get("cachedReadTokens", 0)
    if any(type(n) is not int or n < 0 for n in (prompt, output, cached)) or cached > prompt:
        return None
    usd = (Decimal("0.20") * (prompt - cached) + Decimal("0.04") * cached
           + Decimal("0.50") * output) / 1_000_000
    return {"provider": "Baseten reference only", "billed": False, "currency": "USD",
        "usd": format(usd.normalize(), "f"), "model": model_id,
        "rates_per_million": {"uncached_input": "0.20", "cached_input": "0.04", "output": "0.50"},
        "verified_date": "2026-10-30", "source": "https://www.baseten.co/library/nemotron-35-super-vl/",
        "basis": "NVIDIA usage at Baseten reference rates; cached input is a subset; reasoning is included in output",
        "cache_basis": "reported_cache_read" if "cachedReadTokens" in usage else "missing_cache_charged_uncached",
        "actual_nvidia_cost": "unknown"}


def snapshot(sessions, session_id, profile, effort, *, turn_budget=None):
    root = Path(sessions)
    if root.name != "sessions" or root.parent.name != "pi" or root.is_symlink() or root.parent.is_symlink():
        raise ValueError("Only this trial's pi/sessions receipt directory is allowed")
    matches, other = [], set()
    for path in root.rglob("*.jsonl") if root.exists() else []:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        rows = []
        try:
            for line in path.read_text().split("\n"):
                if line:
                    rows.append(json.loads(line))
        except (ValueError, OSError):
            continue
        if not rows or rows[0].get("type") != "session":
            continue
        sid = rows[0].get("id")
        if sid != session_id:
            other.add(sid)
            continue
        totals = {"inputTokens": 0, "cachedReadTokens": 0, "cachedWriteTokens": 0, "outputTokens": 0}
        reasoning, reasoning_known = 0, True
        seen, calls, models, efforts, error, provider_error = {}, 0, [], [], None, False
        last_assistant, termination_artifact, assistant_count = None, None, 0
        for row in rows[1:]:
            rid = row.get("id")
            if rid:
                if rid in seen:
                    if seen[rid] != row:
                        error = "conflicting_duplicate_event_id"
                    continue
                seen[rid] = row
            if row.get("type") == "model_change":
                models.append((row.get("provider"), row.get("modelId")))
            elif row.get("type") == "thinking_level_change":
                efforts.append(row.get("thinkingLevel"))
            message = row.get("message") or {}
            usage = None
            # Pi 1.4's supported post-batch abort can append a synthetic zero-usage
            # assistant error while unwinding its natural tool continuation. Keep
            # that native record intact, but do not call it inference or a provider
            # failure when the independently persisted 64-turn boundary proves it.
            if (assistant_count == 64 and row is rows[-1] and turn_budget and turn_budget.get("exhausted") is True
                    and turn_budget.get("started") == turn_budget.get("completed") == turn_budget.get("limit") == 64
                    and row.get("type") == "message" and message.get("role") == "assistant"
                    and message.get("stopReason") == "error" and message.get("errorMessage") == "This operation was aborted"
                    and message.get("content") == [] and isinstance(message.get("usage"), dict)
                    and all(message["usage"].get(k) == 0 for k in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"))):
                termination_artifact = {"native_event_id": rid, "reason": "pi_post_budget_abort_zero_usage",
                    "excluded_from_inference_usage": True, "native_record_preserved": True}
                continue
            if row.get("type") in {"compaction", "branch_summary"}:
                usage = row.get("usage")
            elif row.get("type") == "message" and message.get("role") == "assistant":
                assistant_count += 1
                usage = message.get("usage")
                content = message.get("content", [])
                blocks = content if isinstance(content, list) else []
                has_text = bool(content.strip()) if isinstance(content, str) else any(
                    isinstance(b, dict) and b.get("type") == "text" and str(b.get("text", "")).strip() for b in blocks)
                has_tool = any(isinstance(b, dict) and b.get("type") == "toolCall" for b in blocks)
                last_assistant = {"stop_reason": message.get("stopReason"), "has_text": has_text,
                    "has_tool_call": has_tool, "empty_length_stop": message.get("stopReason") == "length"
                    and isinstance(content, (list, str)) and not has_text and not has_tool}
            elif row.get("type") == "message" and message.get("role") == "toolResult" and "usage" in message:
                usage = message["usage"]
            else:
                continue
            if message.get("stopReason") == "error":
                provider_error = True
                error = "native_provider_error"
            metrics = NvidiaPi._metrics_from_usage(usage)
            if metrics is None:
                error = error or "missing_or_unverified_native_usage"
                continue
            totals["inputTokens"] += metrics.prompt_tokens
            totals["cachedReadTokens"] += metrics.cached_tokens
            totals["cachedWriteTokens"] += metrics.extra["cache_write_tokens"]
            totals["outputTokens"] += metrics.completion_tokens
            if type(usage.get("reasoning")) is int and 0 <= usage["reasoning"] <= metrics.completion_tokens:
                reasoning += usage["reasoning"]
            else:
                reasoning_known = False
            calls += 1
        if reasoning_known and calls:
            totals["thoughtTokens"] = reasoning  # Subset of output, not added again.
        if not models or any(m != ("harbor-endpoint", profile.id) for m in models):
            error = "native_model_receipt_mismatch"
        if not efforts or any(e != effort for e in efforts):
            error = "native_thinking_receipt_mismatch"
        matches.append({"status": "verified" if calls and error is None else "unavailable",
            "reason": error or (None if calls else "no_native_usage"), "pricing": "unknown",
            "billed": False, "scope": "selected_root_session_all_calls", "call_count": calls,
            "provider_error": provider_error,
            "last_assistant": last_assistant,
            "termination_artifact": termination_artifact,
            "usage": {**totals, "totalTokens": totals["inputTokens"] + totals["outputTokens"]},
            "native_file": str(path.relative_to(root)),
            "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(matches) != 1:
        return {"status": "unavailable", "reason": "missing_or_ambiguous_native_session", "pricing": "unknown"}
    result = matches[0]
    result["other_native_session_ids"] = sorted(s for s in other if isinstance(s, str))
    if other:
        result.update(status="unavailable", reason="additional_native_sessions_require_reconciliation")
    reference = baseten_reference(result, profile.id)
    if reference is not None:
        result["reference_inference_estimate"] = reference
    return result
