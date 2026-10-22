"Input-token counting, cost reservation and usage settlement."

from __future__ import annotations
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit
import aiohttp

MODEL = "gpt-6-luna"
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-luna"
PRICING_CHECKED = "2026-10-18"
PRICING = {
    "input_per_million_usd": "0.125",
    "output_per_million_usd": "0.50",
    "input_rate_policy": "all input charged at higher cache-write rate",
    "service": "standard non-regional",
    "long_context_threshold": 272000,
}
PRICING_BY_MODEL = {
    "gpt-6-luna": PRICING,
    "gpt-6.1-sol": {
        **PRICING,
        "input_per_million_usd": "2.50",
        "output_per_million_usd": "10.00",
    },
    "gpt-6-astra": {
        **PRICING,
        "input_per_million_usd": "12.50",
        "output_per_million_usd": "50.00",
    },
    "gpt-5.6-sol": {
        **PRICING,
        "input_per_million_usd": "5.00",
        "output_per_million_usd": "20.00",
    },
}
PRICING_SOURCES = {
    model: "https://developers.openai.com/api/docs/models/" + model
    for model in PRICING_BY_MODEL
}


def priced_openai_origin(origin):
    """Only official OpenAI billing or explicit loopback HTTP protocol tests."""
    parsed = urlsplit(origin)
    if origin == "https://api.openai.com":
        return True
    if (
        parsed.scheme not in ("http", "https")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path
    ):
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        return False


def supports_application_spend_cap(config):
    """A pricing-supported route still requires count + reservation per call."""
    from simready_bench.engine.custom_model import supports_custom_budget

    if supports_custom_budget(config):
        return True
    if supports_hf_application_spend_cap(config):
        return True
    budget = config.budget
    return (
        budget is not None
        and budget.pricing_status == "known"
        and config.model in PRICING_BY_MODEL
        and config.api_style == "openai-responses"
        and urlsplit(config.upstream_base_url).path.rstrip("/") == "/v1"
        and priced_openai_origin(config.upstream_origin)
    )


# Provider-native output policy, checked against primary sources 2026-10-18.
# These are reservation bounds, never benchmark-imposed wire limits.
from simready_bench.engine.native_output import AUTHORING_PROTOCOL as NATIVE_OUTPUT_PROTOCOL
NATIVE_OPENAI_MAX_OUTPUT = {"gpt-6-astra": 128000, "gpt-6.1-sol": 128000}
NATIVE_OPENAI_CONTEXT_WINDOW = 1050000
NATIVE_OPENAI_MAX_INPUT = 922000
NATIVE_IDENTITY_UNRESOLVED = frozenset(
    {
        "stepfun-ai/Step-3.7-Flash:deepinfra",
        "zai-org/GLM-4.6V-Flash:novita",
    }
)


def native_output_facts(model):
    """Public facts only. HF context is an upper bound, not an exact output max."""
    from simready_bench.engine.hf_pricing import HF_PRICING, HF_CATALOG_URL

    if model in NATIVE_OPENAI_MAX_OUTPUT:
        bound = NATIVE_OPENAI_MAX_OUTPUT[model]
        return {
            "protocol": NATIVE_OUTPUT_PROTOCOL,
            "maximum_output_tokens": bound,
            "maximum_output_tokens_known": True,
            "output_token_upper_bound": bound,
            "context_window": NATIVE_OPENAI_CONTEXT_WINDOW,
            "maximum_input_tokens": NATIVE_OPENAI_MAX_INPUT,
            "reservation_basis": "documented-provider-maximum-output",
            "source": PRICING_SOURCES[model],
            "checked": "2026-10-18",
            "output_field_policy": "omit",
            "route_identity_status": "resolved",
        }
    if model in HF_PRICING:
        return {
            "protocol": NATIVE_OUTPUT_PROTOCOL,
            "maximum_output_tokens": None,
            "maximum_output_tokens_known": False,
            "context_window": HF_PRICING[model].context_length,
            "maximum_input_tokens": HF_PRICING[model].context_length,
            "output_token_upper_bound": HF_PRICING[model].context_length,
            "reservation_basis": "hf-provider-context-capacity-output-upper-bound",
            "source": HF_CATALOG_URL,
            "checked": "2026-10-18",
            "output_field_policy": "omit",
            "optional_output_field_source": "https://huggingface.co/docs/inference-providers/tasks/chat-completion",
            "route_identity_status": "unresolved"
            if model in NATIVE_IDENTITY_UNRESOLVED
            else "resolved",
        }
    raise CostDenied("native_output_bound_unavailable")


def native_output_reservation_bound(model, *, upstream_origin, api_style):
    facts = native_output_facts(model)
    if facts["route_identity_status"] != "resolved":
        raise CostDenied("native_provider_identity_unresolved")
    if model in NATIVE_OPENAI_MAX_OUTPUT:
        if api_style != "openai-responses" or not priced_openai_origin(upstream_origin):
            raise CostDenied("native_output_route_not_authorized")
    else:
        from simready_bench.engine.hf_pricing import HF_ORIGIN

        if api_style != "openai-completions" or upstream_origin != HF_ORIGIN:
            raise CostDenied("native_output_route_not_authorized")
    return facts["output_token_upper_bound"]


def _native_output_for_payload(payload, *, model, upstream_origin, api_style):
    if any(
        k in payload
        for k in ("max_output_tokens", "max_tokens", "max_completion_tokens")
    ):
        raise CostDenied("native_output_limit_field_forbidden")
    return native_output_reservation_bound(
        model, upstream_origin=upstream_origin, api_style=api_style
    )


COUNT_KEYS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "text",
        "reasoning",
        "parallel_tool_calls",
        "truncation",
    }
)


class CostDenied(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def supports_hf_application_spend_cap(config):
    """Eligibility only: fresh public pricing and reservation still required."""
    from simready_bench.engine.hf_pricing import HF_ORIGIN, HF_PRICING

    budget = config.budget
    return (
        budget is not None
        and budget.pricing_status == "known"
        and config.model in HF_PRICING
        and budget.model == config.model
        and config.api_style == "openai-completions"
        and config.upstream_base_url == HF_ORIGIN + "/v1"
        and budget.upstream_origin == HF_ORIGIN
        and config.upstream_project_id is None
    )


def _hf_input_bound_matches(maximum_input_tokens, context, *, native_output_limits):
    if type(maximum_input_tokens) is not int:
        return False
    # The documented context is an input upper bound whether output is capped
    # per request or left at the provider's native maximum.
    if maximum_input_tokens == context:
        return True
    return not native_output_limits and context <= maximum_input_tokens <= 262144


async def verify_hf_context_bound(
    config, payload, *, maximum_input_tokens, deadline_at
):
    """Use capacity as an upper bound only when the declared limit permits it.

    A smaller input policy requires an actual image-aware counter. This does not
    claim to verify 64k input limits or attribute a token count to a request.
    """
    from simready_bench.engine.hf_pricing import (
        HF_PRICING,
        HFContextEvidence,
        fetch_hf_route,
    )

    budget = config.budget
    if (
        not supports_hf_application_spend_cap(config)
        or budget.project != "simready-bench"
        or budget.approved is not True
        or budget.expires_at <= time.time()
    ):
        raise CostDenied("project_budget_unavailable")
    context = HF_PRICING[config.model].context_length
    if not _hf_input_bound_matches(
        maximum_input_tokens, context,
        native_output_limits=getattr(config, "native_output_limits", False),
    ):
        raise CostDenied("hf_input_limit_unverifiable")
    if not isinstance(payload, dict) or payload.get("model") != config.model:
        raise CostDenied("cost_model_not_authorized")
    route = await fetch_hf_route(
        config.model, deadline_at=min(deadline_at, budget.expires_at)
    )
    return HFContextEvidence(
        config.model, sha(payload), config.upstream_origin, route, time.time()
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def upper_cost_micro_usd(
    input_tokens,
    output_tokens,
    *,
    model=MODEL,
    maximum_input_tokens=256000,
    native_output_limits=False,
):
    output_bound = (
        NATIVE_OPENAI_MAX_OUTPUT.get(model, 4096)
    )
    input_bound = (
        NATIVE_OPENAI_MAX_INPUT
        if model in NATIVE_OPENAI_MAX_OUTPUT
        else 262144
    )
    if (
        type(native_output_limits) is not bool
        or type(maximum_input_tokens) is not int
        or not 1 <= maximum_input_tokens <= input_bound
        or any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens))
        or input_tokens > maximum_input_tokens
        or output_tokens > output_bound
    ):
        raise CostDenied("unsupported_cost_bounds")
    if model not in PRICING_BY_MODEL:
        raise CostDenied("cost_model_not_authorized")
    schedule = PRICING_BY_MODEL[model]
    # Per-million dollar rates equal per-token micro-dollar rates. Ceil once,
    # so concurrent requests cannot exploit fractional rounding or floats.
    long_context = input_tokens > schedule["long_context_threshold"]
    input_multiplier = Decimal(2) if long_context else Decimal(1)
    output_multiplier = Decimal("1.5") if long_context else Decimal(1)
    return int(
        (
            Decimal(input_tokens)
            * Decimal(schedule["input_per_million_usd"])
            * input_multiplier
            + Decimal(output_tokens)
            * Decimal(schedule["output_per_million_usd"])
            * output_multiplier
        ).to_integral_value(rounding=ROUND_CEILING)
    )


@dataclass(frozen=True)
class InputTokenEvidence:
    model: str
    input_tokens: int
    request_sha256: str
    counted_payload_sha256: str
    counter_origin: str
    verified_at: float
    source: str = "openai-responses-input-tokens"
    counter_project_sha256: str | None = None


async def count_input_tokens(config, payload, *, maximum_input_tokens, deadline_at):
    """Actual fixed provider counter; no authority to generate a model response.

    Caller must run the authoritative gateway _budget() before this function.
    This independently rechecks the same approved/expires/cap preconditions so
    direct trusted callers also fail closed while budget is absent.
    """
    budget = config.budget
    if (
        budget is None
        or budget.project != "simready-bench"
        or budget.approved is not True
        or not supports_application_spend_cap(config)
        or budget.expires_at <= time.time()
        or budget.model != config.model
        or config.model not in PRICING_BY_MODEL
        or budget.upstream_origin != config.upstream_origin
    ):
        raise CostDenied("project_budget_unavailable")
    if config.api_style != "openai-responses":
        raise CostDenied("verified_input_count_requires_responses")
    native = getattr(config, "native_output_limits", False)
    provider_input_bound = (
        native_output_facts(config.model)["maximum_input_tokens"]
    )
    if (
        type(maximum_input_tokens) is not int
        or not 1 <= maximum_input_tokens <= provider_input_bound
        or (native and maximum_input_tokens != provider_input_bound)
    ):
        raise CostDenied("unsupported_input_limit")
    if not isinstance(payload, dict) or payload.get("model") != config.model:
        raise CostDenied("cost_model_not_authorized")
    counted = {k: v for k, v in payload.items() if k in COUNT_KEYS}
    if "input" not in counted:
        raise CostDenied("countable_input_required")
    duration = min(30, deadline_at - time.time(), budget.expires_at - time.time())
    if duration <= 0:
        raise CostDenied("input_counter_deadline_exceeded")
    endpoint = config.upstream_base_url.rstrip("/") + "/responses/input_tokens"
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Authorization": "Bearer " + config.upstream_api_key,
    }
    if config.upstream_project_id is not None:
        headers["OpenAI-Project"] = config.upstream_project_id
    try:
        async with aiohttp.ClientSession(
            trust_env=False,
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=duration),
        ) as session:
            async with session.post(
                endpoint,
                data=canonical(counted).encode(),
                allow_redirects=False,
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise CostDenied("input_counter_unavailable")
                chunks = []
                size = 0
                async for chunk in response.content.iter_chunked(16384):
                    size += len(chunk)
                    if size > 65536:
                        raise CostDenied("input_counter_response_limit")
                    chunks.append(chunk)
                result = json.loads(b"".join(chunks))
    except CostDenied:
        raise
    except (aiohttp.ClientError, TimeoutError, ValueError):
        raise CostDenied("input_counter_unavailable") from None
    tokens = result.get("input_tokens")
    if (
        result.get("object") != "response.input_tokens"
        or type(tokens) is not int
        or tokens < 0
    ):
        raise CostDenied("input_counter_response_invalid")
    if tokens > maximum_input_tokens:
        raise CostDenied("input_token_budget_exceeded")
    return InputTokenEvidence(
        config.model,
        tokens,
        sha(payload),
        sha(counted),
        config.upstream_origin,
        time.time(),
        counter_project_sha256=hashlib.sha256(
            config.upstream_project_id.encode()
        ).hexdigest()
        if config.upstream_project_id is not None
        else None,
    )


class CostLedger:
    """One atomic durable reservation ledger shared by actor and judge calls."""

    def __init__(self, state_dir):
        self.path = Path(state_dir) / "request-cost.sqlite3"
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reservations(request_id TEXT PRIMARY KEY,authorization_id TEXT NOT NULL,fingerprint TEXT NOT NULL,micro_usd INTEGER NOT NULL,purpose TEXT NOT NULL,record TEXT NOT NULL)"
            )
        self.path.chmod(0o600)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cost_settlements(request_id TEXT PRIMARY KEY,settled_upper_micro_usd INTEGER NOT NULL CHECK(settled_upper_micro_usd>=0),proof_sha256 TEXT NOT NULL,proof_json TEXT NOT NULL,created_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS cost_overruns(request_id TEXT PRIMARY KEY,excess_micro_usd INTEGER NOT NULL CHECK(excess_micro_usd>0),proof_sha256 TEXT NOT NULL,proof_json TEXT NOT NULL,created_at REAL NOT NULL)"
            )
            for table in ("cost_settlements", "cost_overruns", "reservations"):
                for operation in ("UPDATE", "DELETE"):
                    db.execute(
                        f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT,'append_only_cost_history'); END"
                    )

    def settle(self, request_id, *, evidence, cost_payload_sha256=None):
        with sqlite3.connect(self.path, timeout=10, isolation_level=None) as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT micro_usd,record FROM reservations WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if row is None:
                    raise CostDenied("cost_reservation_unknown")
                original = json.loads(row[1])
                result = verify_usage_settlement(
                    evidence,
                    request_id=request_id,
                    original_upper_micro_usd=row[0],
                    allow_overrun=True,
                )
                proof = result["proof"]
                encoded = canonical(proof)
                if (
                    cost_payload_sha256
                    if cost_payload_sha256 is not None
                    else proof["request_sha256"]
                ) != original["request_sha256"] or proof["model"] != original["model"]:
                    raise CostDenied("cost_settlement_original_request_mismatch")
                if ((original.get("route_metadata_sha256") is not None
                     or proof.get("route_metadata_sha256") is not None)
                        and (original.get("route_metadata_sha256") is None
                             or proof.get("route_metadata_sha256") != original["route_metadata_sha256"])):
                    raise CostDenied("cost_settlement_original_route_mismatch")
                overrun = result["settled_upper_micro_usd"] > row[0]
                table = "cost_overruns" if overrun else "cost_settlements"
                value = (
                    result["settled_upper_micro_usd"] - row[0]
                    if overrun
                    else result["settled_upper_micro_usd"]
                )
                old = db.execute(
                    f"SELECT proof_sha256,proof_json FROM {table} WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if old is not None and old != (result["proof_sha256"], encoded):
                    raise CostDenied("cost_settlement_conflict")
                if old is None:
                    db.execute(
                        f"INSERT INTO {table} VALUES (?,?,?,?,?)",
                        (
                            request_id,
                            value,
                            result["proof_sha256"],
                            encoded,
                            time.time(),
                        ),
                    )
                db.commit()
            except BaseException:
                db.rollback()
                raise
        if overrun:
            raise CostDenied("cost_usage_overrun_spending_blocked")
        return {**result, "idempotent": old is not None}

    def reserve(
        self,
        *,
        budget,
        payload,
        evidence,
        request_id,
        purpose,
        upstream_project_id=None,
        maximum_input_tokens=None,
        native_output_limits=False,
    ):
        from simready_bench.engine.hf_pricing import HFContextEvidence

        if type(evidence) is HFContextEvidence:
            return self._reserve_hf(
                budget=budget,
                payload=payload,
                evidence=evidence,
                request_id=request_id,
                purpose=purpose,
                upstream_project_id=upstream_project_id,
                maximum_input_tokens=maximum_input_tokens,
                native_output_limits=native_output_limits,
            )
        if type(evidence) is not InputTokenEvidence or purpose not in (
            "actor",
            "judge",
        ):
            raise CostDenied("verified_input_count_required")
        if type(native_output_limits) is not bool:
            raise CostDenied("native_output_policy_invalid")
        provider_input_bound = (
            native_output_facts(budget.model)["maximum_input_tokens"]
        )
        maximum = (
            maximum_input_tokens
            if maximum_input_tokens is not None
            else (
                provider_input_bound
                if native_output_limits
                else (64000 if purpose == "actor" else 256000)
            )
        )
        if (
            type(maximum) is not int
            or not 1 <= maximum <= provider_input_bound
            or (native_output_limits and maximum != provider_input_bound)
        ):
            raise CostDenied("unsupported_input_limit")
        project_sha = (
            hashlib.sha256(upstream_project_id.encode()).hexdigest()
            if upstream_project_id is not None
            else None
        )
        if evidence.counter_project_sha256 != project_sha:
            raise CostDenied("verified_input_count_required")
        if (
            budget.project != "simready-bench"
            or budget.model not in PRICING_BY_MODEL
            or budget.approved is not True
            or budget.pricing_status != "known"
            or not priced_openai_origin(budget.upstream_origin)
            or budget.expires_at <= time.time()
            or evidence.model != budget.model
            or evidence.counter_origin != budget.upstream_origin
            or evidence.request_sha256 != sha(payload)
            or time.time() - evidence.verified_at > 30
            or type(evidence.verified_at) not in (int, float)
            or not math.isfinite(evidence.verified_at)
            or evidence.verified_at > time.time() + 1
            or evidence.input_tokens > maximum
        ):
            raise CostDenied("verified_input_count_required")
        if type(native_output_limits) is not bool:
            raise CostDenied("native_output_policy_invalid")
        output = (
            _native_output_for_payload(
                payload,
                model=budget.model,
                upstream_origin=budget.upstream_origin,
                api_style="openai-responses",
            )
            if native_output_limits
            else payload.get("max_output_tokens")
        )
        if type(output) is not int or not 1 <= output <= (
            NATIVE_OPENAI_MAX_OUTPUT.get(budget.model, 0)
        ):
            raise CostDenied("output_token_budget_required")
        amount = upper_cost_micro_usd(
            evidence.input_tokens,
            output,
            model=budget.model,
            maximum_input_tokens=maximum,
            native_output_limits=native_output_limits,
        )
        cap = Decimal(str(budget.maximum_spend_usd)) * Decimal(1000000)
        if not cap.is_finite() or cap <= 0:
            raise CostDenied("project_budget_unavailable")
        identity = {
            "budget": asdict(budget),
            "pricing": PRICING_BY_MODEL[budget.model],
            "checked": PRICING_CHECKED,
        }
        if native_output_limits:
            identity["native_output_policy"] = native_output_facts(budget.model)
        if maximum_input_tokens is not None:
            identity["maximum_input_tokens"] = maximum_input_tokens
        if project_sha is not None:
            identity["upstream_project_sha256"] = project_sha
        fingerprint = sha(identity)
        record = {
            "purpose": purpose,
            "input_tokens": evidence.input_tokens,
            "max_output_tokens": output,
            "request_sha256": evidence.request_sha256,
            "reserved_upper_cost_micro_usd": amount,
            "actual_cost_usd": None,
            "pricing_source": PRICING_SOURCES[budget.model],
            "model": budget.model,
            "pricing_checked": PRICING_CHECKED,
        }
        if native_output_limits:
            record.update(native_output_facts(budget.model))
            record["max_output_tokens"] = None
            record["long_context_pricing_applied"] = (
                evidence.input_tokens
                > PRICING_BY_MODEL[budget.model]["long_context_threshold"]
            )
        return self._reserve_amount(
            budget, request_id, purpose, amount, fingerprint, record
        )

    def _reserve_hf(
        self,
        *,
        budget,
        payload,
        evidence,
        request_id,
        purpose,
        upstream_project_id,
        maximum_input_tokens,
        native_output_limits=False,
    ):
        from simready_bench.engine.hf_pricing import (
            HF_ORIGIN,
            HF_CATALOG_URL,
            HF_PRICING,
            HF_PRICING_CHECKED,
            HFRouteEvidence,
            hf_upper_cost_micro_usd,
        )

        now = time.time()
        route = evidence.route
        if (
            type(route) is not HFRouteEvidence
            or purpose not in ("actor", "judge")
            or upstream_project_id is not None
            or budget.project != "simready-bench"
            or budget.model not in HF_PRICING
            or budget.approved is not True
            or budget.pricing_status != "known"
            or budget.upstream_origin != HF_ORIGIN
            or budget.expires_at <= now
            or evidence.model != budget.model
            or evidence.upstream_origin != HF_ORIGIN
            or payload.get("model") != budget.model
            or evidence.request_sha256 != sha(payload)
            or route.model != budget.model
            or route.pricing != HF_PRICING[budget.model]
            or route.source != HF_CATALOG_URL
            or evidence.source != "hf-provider-context-capacity"
            or any(
                type(t) not in (int, float)
                or not math.isfinite(t)
                or not now - 30 <= t <= now + 1
                for t in (evidence.verified_at, route.verified_at)
            )
        ):
            raise CostDenied("hf_context_evidence_required")
        context = route.pricing.context_length
        if not _hf_input_bound_matches(
            maximum_input_tokens, context, native_output_limits=native_output_limits,
        ):
            raise CostDenied("hf_input_limit_unverifiable")
        if type(native_output_limits) is not bool:
            raise CostDenied("native_output_policy_invalid")
        output = (
            _native_output_for_payload(
                payload,
                model=budget.model,
                upstream_origin=budget.upstream_origin,
                api_style="openai-completions",
            )
            if native_output_limits
            else payload.get("max_tokens")
        )
        if (
            "max_completion_tokens" in payload
            or "max_output_tokens" in payload
            or type(payload.get("n", 1)) is not int
            or payload.get("n", 1) != 1
        ):
            raise CostDenied("unsupported_cost_bounds")
        amount = hf_upper_cost_micro_usd(
            budget.model, output, native_output_limits=native_output_limits
        )
        identity = {
            "budget": asdict(budget),
            "pricing": asdict(route.pricing),
            "checked": HF_PRICING_CHECKED,
            "maximum_input_tokens": maximum_input_tokens,
            "route_metadata_sha256": route.metadata_sha256,
        }
        if native_output_limits:
            identity["native_output_policy"] = native_output_facts(budget.model)
        record = {
            "purpose": purpose,
            "model": budget.model,
            "input_tokens": None,
            "input_count_verified": False,
            "input_token_upper_bound": context,
            "declared_maximum_input_tokens": maximum_input_tokens,
            "max_output_tokens": output,
            "request_sha256": evidence.request_sha256,
            "reserved_upper_cost_micro_usd": amount,
            "actual_cost_usd": None,
            "pricing_source": HF_CATALOG_URL,
            "pricing_checked": HF_PRICING_CHECKED,
            "route_metadata_sha256": route.metadata_sha256,
            "reservation_basis": "hf-provider-context-capacity",
        }
        if native_output_limits:
            record.update(native_output_facts(budget.model))
            record["max_output_tokens"] = None
        return self._reserve_amount(
            budget, request_id, purpose, amount, sha(identity), record
        )

    def _reserve_amount(self, budget, request_id, purpose, amount, fingerprint, record):
        cap = Decimal(str(budget.maximum_spend_usd)) * Decimal(1000000)
        if not cap.is_finite() or cap <= 0:
            raise CostDenied("project_budget_unavailable")
        with sqlite3.connect(self.path, timeout=10, isolation_level=None) as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT COUNT(*) FROM cost_overruns").fetchone()[0]:
                    raise CostDenied("cost_usage_overrun_spending_blocked")
                rows = db.execute(
                    "SELECT r.fingerprint,COALESCE(s.settled_upper_micro_usd,r.micro_usd) FROM reservations r LEFT JOIN cost_settlements s USING(request_id) WHERE r.authorization_id=?",
                    (budget.authorization_id,),
                ).fetchall()
                if any(row[0] != fingerprint for row in rows):
                    raise CostDenied("budget_authorization_conflict")
                if sum(row[1] for row in rows) + amount > cap:
                    raise CostDenied("project_spend_budget_exhausted")
                try:
                    db.execute(
                        "INSERT INTO reservations VALUES (?,?,?,?,?,?)",
                        (
                            request_id,
                            budget.authorization_id,
                            fingerprint,
                            amount,
                            purpose,
                            canonical(record),
                        ),
                    )
                except sqlite3.IntegrityError:
                    raise CostDenied("request_cost_admission_reused") from None
                db.commit()
            except BaseException:
                db.rollback()
                raise
        # Never refund uncertain/provider-error attempts: retained reservations
        # conservatively include retries and lost resources.
        return record


# Reconciliation is a trusted-host observer operation, never an actor HTTP API.
# Returned token usage permits a conservative priced bound even when cache
# details/invoices are absent. Original admission bounds remain immutable.
@dataclass(frozen=True)
class UsageSettlementEvidence:
    record_json: str


_SETTLEMENT_RECORD_KEYS = frozenset(
    {
        "gateway_request_id",
        "model",
        "provider_reported_models",
        "api_path",
        "status",
        "upstream_status",
        "request_sha256",
        "response_sha256",
        "pricing_checked",
        "pricing_source",
        "route_metadata_sha256",
        "max_output_tokens",
        "maximum_output_tokens",
        "output_token_upper_bound",
        "reserved_upper_cost_micro_usd",
        "usage",
    }
)


def make_usage_settlement_evidence(record):
    """Project a trusted gateway observer record; do not pass actor JSON here."""
    if type(record) is not dict:
        raise CostDenied("trusted_usage_observer_record_required")
    selected = {k: v for k, v in record.items() if k in _SETTLEMENT_RECORD_KEYS}
    usage = selected.get("usage")
    if type(usage) is dict:
        selected["usage"] = {
            k: v
            for k, v in usage.items()
            if k
            in {
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens_details",
                "output_tokens_details",
                "prompt_tokens_details",
                "completion_tokens_details",
            }
        }
    try:
        raw = canonical(selected)
    except (TypeError, ValueError):
        raise CostDenied("invalid_usage_observer_record") from None
    if len(raw.encode()) > 65536:
        raise CostDenied("usage_observer_record_limit")
    return UsageSettlementEvidence(raw)


def verify_usage_settlement(
    evidence, *, request_id, original_upper_micro_usd, allow_overrun=False
):
    """Validate recorded usage and calculate the cost of its reserved request."""
    from simready_bench.engine.hf_pricing import (
        HF_PRICING,
        HF_CATALOG_URL,
        HF_PRICING_CHECKED,
    )
    import re

    if type(allow_overrun) is not bool:
        raise CostDenied("invalid_usage_overrun_policy")
    if (
        type(evidence) is not UsageSettlementEvidence
        or type(evidence.record_json) is not str
    ):
        raise CostDenied("trusted_usage_observer_record_required")
    if len(evidence.record_json.encode()) > 65536:
        raise CostDenied("usage_observer_record_limit")
    try:
        r = json.loads(evidence.record_json)
        if type(r) is not dict or canonical(r) != evidence.record_json:
            raise ValueError("noncanonical observer record")
    except (TypeError, ValueError):
        raise CostDenied("invalid_usage_observer_record") from None
    if (
        type(original_upper_micro_usd) is not int
        or original_upper_micro_usd <= 0
        or r.get("gateway_request_id") != request_id
        or r.get("reserved_upper_cost_micro_usd") != original_upper_micro_usd
    ):
        raise CostDenied("usage_original_attempt_binding_required")
    if r.get("status") != "completed" or r.get("upstream_status") != 200:
        raise CostDenied("completed_provider_usage_required")
    for key in ("request_sha256", "response_sha256"):
        if type(r.get(key)) is not str or not re.fullmatch(r"[0-9a-f]{64}", r[key]):
            raise CostDenied("usage_response_request_hash_required")
    model = r.get("model")
    hf = model in HF_PRICING
    if model not in PRICING_BY_MODEL and not hf:
        raise CostDenied("usage_pricing_unavailable")
    expected = model.rsplit(":", 1)[0] if hf else model
    reported = r.get("provider_reported_models")
    if (
        type(reported) is not list
        or not 1 <= len(reported) <= 16
        or any(
            type(m) is not str or m.casefold() != expected.casefold() for m in reported
        )
    ):
        raise CostDenied("usage_provider_identity_unresolved")
    source = HF_CATALOG_URL if hf else PRICING_SOURCES[model]
    checked = HF_PRICING_CHECKED if hf else PRICING_CHECKED
    if (
        r.get("pricing_source") != source
        or r.get("pricing_checked") != checked
        or r.get("api_path") != ("/v1/chat/completions" if hf else "/v1/responses")
    ):
        raise CostDenied("usage_pricing_binding_required")
    if hf and (
        type(r.get("route_metadata_sha256")) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", r["route_metadata_sha256"])
    ):
        raise CostDenied("usage_hf_route_pin_required")
    usage = r.get("usage")
    if type(usage) is not dict:
        raise CostDenied("complete_integer_provider_usage_required")
    i = usage.get("prompt_tokens" if hf else "input_tokens")
    o = usage.get("completion_tokens" if hf else "output_tokens")
    total = usage.get("total_tokens")
    if any(type(n) is not int or n < 0 for n in (i, o, total)) or total != i + o:
        raise CostDenied("complete_integer_provider_usage_required")
    output_bound = r.get("output_token_upper_bound")
    if output_bound is None:
        output_bound = r.get("max_output_tokens")
    if output_bound is None:
        output_bound = r.get("maximum_output_tokens")
    if type(output_bound) is not int or output_bound <= 0 or o > output_bound:
        raise CostDenied("usage_output_exceeds_admitted_bound")
    if hf:
        pin = HF_PRICING[model]
        if i > pin.context_length:
            raise CostDenied("usage_input_exceeds_provider_bound")
        ip = Decimal(pin.input_per_million_usd)
        op = Decimal(pin.output_per_million_usd)
        amount = int(
            (Decimal(i) * ip + Decimal(o) * op).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        pricing = {
            "input_per_million_usd": str(ip),
            "output_per_million_usd": str(op),
            "cache_detail_policy": "all input charged at full pinned input rate",
        }
    else:
        native = model in NATIVE_OPENAI_MAX_OUTPUT
        amount = upper_cost_micro_usd(
            i,
            o,
            model=model,
            maximum_input_tokens=NATIVE_OPENAI_MAX_INPUT if native else 262144,
            native_output_limits=native,
        )
        schedule = PRICING_BY_MODEL[model]
        long = i > schedule["long_context_threshold"]
        pricing = {
            "input_per_million_usd": schedule["input_per_million_usd"],
            "output_per_million_usd": schedule["output_per_million_usd"],
            "cache_detail_policy": "Highest pinned input cache-write rate",
            "long_context_pricing_applied": long,
            "input_multiplier": 2 if long else 1,
            "output_multiplier": "1.5" if long else "1",
        }
    if amount > original_upper_micro_usd and not allow_overrun:
        raise CostDenied("usage_cost_exceeds_original_hold")
    proof = {
        "schema": "provider-usage-priced-settlement/v1",
        "request_id": request_id,
        "original_upper_micro_usd": original_upper_micro_usd,
        "settled_upper_micro_usd": amount,
        "unused_upper_micro_usd": max(0, original_upper_micro_usd - amount),
        "cost_basis": "provider_usage_priced_upper_bound",
        "invoice_verified": False,
        "model": model,
        "provider_reported_models": reported,
        "input_tokens": i,
        "output_tokens": o,
        "request_sha256": r["request_sha256"],
        "response_sha256": r["response_sha256"],
        "gateway_observer_record_sha256": sha(r),
        "pricing_source": source,
        "pricing_checked": checked,
        "pricing": pricing,
        "route_metadata_sha256": r.get("route_metadata_sha256"),
    }
    if amount > original_upper_micro_usd:
        proof["overrun_micro_usd"] = amount - original_upper_micro_usd
    return {
        "settled_upper_micro_usd": amount,
        "overrun_micro_usd": max(0, amount - original_upper_micro_usd),
        "cost_basis": proof["cost_basis"],
        "invoice_verified": False,
        "proof": proof,
        "proof_sha256": sha(proof),
    }
