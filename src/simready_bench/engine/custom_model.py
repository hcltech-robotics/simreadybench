"Operator-supplied model capacity and pricing metadata."

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from urllib.parse import urlsplit
import ipaddress, math, re, time


def validate_hf_router_relay(value, *, base_url, model=None):
    """Validate the private route retained by the owned Daytona relay."""
    from simready_bench.engine.hf_pricing import HF_ORIGIN

    if (type(value) is not dict or set(value) != {
            "schema", "model", "upstream_base_url", "relay_base_url"}
            or value.get("schema") != "simready-hf-router-relay/v1"
            or value.get("upstream_base_url") != HF_ORIGIN + "/v1"
            or value.get("relay_base_url") != base_url
            or not isinstance(value.get("model"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}", value["model"])
            or (model is not None and value["model"] != model)):
        raise ValueError("Exact HF Router relay route binding required")
    endpoint = urlsplit(base_url)
    try:
        loopback = endpoint.hostname == "localhost" or ipaddress.ip_address(endpoint.hostname or "").is_loopback
        port = endpoint.port
    except ValueError:
        loopback, port = False, None
    if (endpoint.scheme != "http" or not loopback or not port
            or endpoint.path != "/v1" or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment):
        raise ValueError("Exact loopback HF Router relay endpoint required")
    return deepcopy(value)


def _uses_hf_router(config):
    from simready_bench.engine.hf_pricing import HF_ORIGIN

    binding = getattr(config, "hf_router_relay", None)
    if binding is not None:
        validate_hf_router_relay(binding, base_url=config.upstream_base_url, model=config.model)
        return True
    return config.upstream_base_url.rstrip("/") == HF_ORIGIN + "/v1"


def _validated_hf_route(config, route):
    from simready_bench.engine.hf_pricing import HF_CATALOG_URL, HF_PRICING, HFRouteEvidence
    from simready_bench.engine.request_cost import CostDenied

    if not _uses_hf_router(config):
        if route is not None:
            raise CostDenied("hf_model_route_binding_required")
        return None
    if config.model not in HF_PRICING:
        raise CostDenied("hf_cost_model_not_authorized")
    expected = HF_PRICING[config.model]
    if (type(route) is not HFRouteEvidence or route.model != config.model
            or route.source != HF_CATALOG_URL or route.pricing != expected
            or type(route.metadata_sha256) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", route.metadata_sha256)
            or type(route.verified_at) not in (int, float)
            or not math.isfinite(route.verified_at)
            or not time.time() - 30 <= route.verified_at <= time.time() + 1):
        raise CostDenied("hf_model_route_evidence_required")
    spec = validate_model_spec(config.model_spec, model=config.model,
                               base_url=config.upstream_base_url, api_style=config.api_style)
    if (config.api_style != "openai-completions"
            or spec["context_window"] != expected.context_length
            or Decimal(str(spec["pricing"]["input_per_million_usd"])) != Decimal(expected.input_per_million_usd)
            or Decimal(str(spec["pricing"]["output_per_million_usd"])) != Decimal(expected.output_per_million_usd)
            or (config.native_output_limits and spec["maximum_output_tokens"] not in (None, expected.context_length))):
        raise CostDenied("hf_model_declaration_route_mismatch")
    return route


async def verify_custom_hf_route(config, *, deadline_at):
    """Verify catalogue declarations only for a genuine direct or private HF route."""
    if not _uses_hf_router(config):
        return None
    from simready_bench.engine.hf_pricing import fetch_hf_route

    route = await fetch_hf_route(config.model, deadline_at=min(deadline_at, config.budget.expires_at))
    return _validated_hf_route(config, route)


def validate_model_spec(value, *, model=None, base_url=None, api_style=None):
    fields = {
        "schema",
        "model",
        "base_url",
        "api_style",
        "context_window",
        "maximum_output_tokens",
        "pricing",
        "pricing_source",
        "checked",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema"] != "simready-operator-model-v1"
    ):
        raise ValueError("Exact operator model declaration required")
    v = deepcopy(value)
    if not isinstance(v["model"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}", v["model"]
    ):
        raise ValueError("Literal model identity required")
    for k in ("base_url", "pricing_source"):
        if not isinstance(v[k], str):
            raise ValueError("Literal model/pricing URL required")
        u = urlsplit(v[k])
        loopback = u.hostname in ("localhost", "127.0.0.1", "::1")
        if (
            not u.hostname
            or u.username
            or u.password
            or u.query
            or u.fragment
            or (u.scheme != "https" and not (u.scheme == "http" and loopback))
        ):
            raise ValueError("Fixed HTTPS model/pricing route required")
    if v["api_style"] not in ("openai-responses", "openai-completions"):
        raise ValueError("Compatible wire protocol required")
    if type(v["context_window"]) is not int or not 1 <= v["context_window"] <= 10**9:
        raise ValueError("Provider context capacity required")
    if v["maximum_output_tokens"] is not None and (
        type(v["maximum_output_tokens"]) is not int
        or not 1 <= v["maximum_output_tokens"] <= v["context_window"]
    ):
        raise ValueError("Output capacity must fit provider context")
    if type(v["pricing"]) is not dict or set(v["pricing"]) != {
        "input_per_million_usd",
        "output_per_million_usd",
    }:
        raise ValueError("Explicit conservative uncached token pricing required")
    for rate in v["pricing"].values():
        try:
            d = Decimal(str(rate))
        except (InvalidOperation, ValueError):
            raise ValueError("Finite token rate required") from None
        if not d.is_finite() or d < 0 or d > 10**9:
            raise ValueError("Finite nonnegative token rate required")
    if not isinstance(v["checked"], str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", v["checked"]
    ):
        raise ValueError("Pricing declaration date required")
    if any(
        expected is not None and v[k] != expected
        for k, expected in (
            ("model", model),
            ("base_url", base_url),
            ("api_style", api_style),
        )
    ):
        raise ValueError("Operator model route binding differs")
    return v


def facts(spec):
    from simready_bench.engine.native_output import AUTHORING_PROTOCOL

    v = validate_model_spec(spec)
    bound = v["maximum_output_tokens"] or v["context_window"]
    return {
        "protocol": AUTHORING_PROTOCOL,
        "maximum_output_tokens": v["maximum_output_tokens"],
        "maximum_output_tokens_known": v["maximum_output_tokens"] is not None,
        "output_token_upper_bound": bound,
        "context_window": v["context_window"],
        "maximum_input_tokens": v["context_window"],
        "reservation_basis": "operator-declared-provider-capacity",
        "source": v["pricing_source"],
        "checked": v["checked"],
        "output_field_policy": "omit",
        "route_identity_status": "operator_declared",
        "declaration_status": "operator_declared_not_provider_verified",
    }


def configured_facts(config):
    if getattr(config, "model_spec", None) is not None:
        return facts(config.model_spec)
    from simready_bench.engine.request_cost import native_output_facts

    return native_output_facts(config.model)


def supports_custom_budget(config):
    if getattr(config, "model_spec", None) is None:
        return False
    try:
        validate_model_spec(
            config.model_spec,
            model=config.model,
            base_url=config.upstream_base_url,
            api_style=config.api_style,
        )
    except ValueError:
        return False
    b = config.budget
    return (
        b is not None
        and b.approved is True
        and b.pricing_status == "known"
        and b.model == config.model
        and b.project == "simready-bench"
        and b.upstream_origin == config.upstream_origin
        and b.expires_at > time.time()
    )


def reserve_custom_cost(config, payload, ledger, request_id, purpose, *, route=None):
    from simready_bench.engine.request_cost import CostDenied, sha

    if (
        not supports_custom_budget(config)
        or payload.get("model") != config.model
        or purpose not in ("actor", "judge")
    ):
        raise CostDenied("custom_model_authority_unavailable")
    spec = validate_model_spec(config.model_spec)
    route = _validated_hf_route(config, route)
    f = facts(spec)
    if config.native_output_limits:
        if any(
            k in payload
            for k in ("max_output_tokens", "max_tokens", "max_completion_tokens")
        ):
            raise CostDenied("native_output_limit_field_forbidden")
        output = f["output_token_upper_bound"]
    else:
        output = (
            payload.get("max_output_tokens")
            if config.api_style == "openai-responses"
            else payload.get("max_completion_tokens", payload.get("max_tokens"))
        )
        if type(output) is not int or not 1 <= output <= f["output_token_upper_bound"]:
            raise CostDenied("custom_model_output_bound_required")
    maximum = spec["context_window"]
    rates = spec["pricing"]
    amount = int(
        (
            Decimal(maximum) * Decimal(str(rates["input_per_million_usd"]))
            + Decimal(output) * Decimal(str(rates["output_per_million_usd"]))
        ).to_integral_value(rounding=ROUND_CEILING)
    )
    record = {
        "purpose": purpose,
        "model": config.model,
        "input_tokens": None,
        "input_count_verified": False,
        "input_token_upper_bound": maximum,
        "max_output_tokens": None if config.native_output_limits else output,
        "request_sha256": sha(payload),
        "reserved_upper_cost_micro_usd": amount,
        "actual_cost_usd": None,
        "pricing_source": spec["pricing_source"],
        "pricing_checked": spec["checked"],
        "model_spec_sha256": sha(spec),
        "reservation_basis": "operator-declared-provider-capacity",
        "declaration_status": "operator_declared_not_provider_verified",
    }
    if config.native_output_limits:
        record.update(f)
    identity = {"budget": asdict(config.budget), "model_spec": spec}
    if route is not None:
        from simready_bench.engine.hf_pricing import HF_CATALOG_URL, HF_PRICING_CHECKED, HF_ORIGIN

        proof = {"model": route.model, "upstream_base_url": HF_ORIGIN + "/v1",
                 "pricing": asdict(route.pricing), "source": route.source,
                 "metadata_sha256": route.metadata_sha256}
        identity["hf_router_route"] = proof
        record.update(route_metadata_sha256=route.metadata_sha256,
                      route_verified_at=route.verified_at, hf_router_route=proof,
                      pricing_source=HF_CATALOG_URL, pricing_checked=HF_PRICING_CHECKED,
                      reservation_basis=("hf-provider-context-capacity-output-upper-bound"
                                         if config.native_output_limits else "hf-provider-context-capacity"),
                      source=HF_CATALOG_URL, checked=HF_PRICING_CHECKED,
                      route_identity_status="hf_router_catalogue_verified",
                      declaration_status="matched_hf_router_catalogue")
    return ledger._reserve_amount(
        config.budget,
        request_id,
        purpose,
        amount,
        sha(identity),
        record,
    )
