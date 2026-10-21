"""Trusted HF price pins, independent of Harbor's native model discovery.

A provider context capacity is a monetary upper bound, NOT a token count.
There is no documented HF Router image-aware, pre-generation token counter.
The full capacity is reserved even for tiny prompts and failed/uncertain calls.
Public rate metadata is revalidated before every admission, without credentials.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
import math
import random
import time
from types import MappingProxyType
from weakref import WeakKeyDictionary

import aiohttp

HF_ORIGIN = "https://router.huggingface.co"
HF_CATALOG_URL = HF_ORIGIN + "/v1/models"
HF_PRICING_CHECKED = "2026-10-18"
_INFLIGHT = WeakKeyDictionary()


@dataclass(frozen=True)
class HFPrice:
    context_length: int
    input_per_million_usd: str
    output_per_million_usd: str


# Explicit provider suffixes prohibit cheapest/fastest/automatic routing.
HF_PRICING = MappingProxyType(
    {
        "Qwen/Qwen3.8-27B:deepinfra": HFPrice(262144, "0.20", "2.50"),
        "google/gemma-4-31B-it:novita": HFPrice(262144, "0.14", "0.40"),
        "meta-models/Muse-Glimmer-30B:deepinfra": HFPrice(131072, "0.30", "1.20"),
        "meta-models/Muse-Spark-Open-816B-A42B:baseten": HFPrice(1048576, "1.25", "4.25"),
        "zai-org/GLM-4.6V-Flash:novita": HFPrice(131072, "0.30", "0.90"),
        "stepfun-ai/Step-3.7-Flash:deepinfra": HFPrice(262144, "0.20", "1.15"),
    }
)


@dataclass(frozen=True)
class HFRouteEvidence:
    model: str
    pricing: HFPrice
    metadata_sha256: str
    verified_at: float
    source: str = HF_CATALOG_URL


@dataclass(frozen=True)
class HFContextEvidence:
    model: str
    request_sha256: str
    upstream_origin: str
    route: HFRouteEvidence
    verified_at: float
    source: str = "hf-provider-context-capacity"


def _deny(code):
    from simready_bench.engine.request_cost import CostDenied

    raise CostDenied(code)


def validate_catalog_route(catalog, model):
    """Reject missing/ambiguous routes, changed rates/capacity, or lost features."""
    if model not in HF_PRICING:
        _deny("hf_cost_model_not_authorized")
    base, provider = model.rsplit(":", 1)
    expected = HF_PRICING[model]
    try:
        rows = [r for r in catalog["data"] if r["id"] == base]
        if len(rows) != 1:
            _deny("hf_pricing_route_unavailable")
        row = rows[0]
        routes = [r for r in row["providers"] if r["provider"] == provider]
        if len(routes) != 1:
            _deny("hf_pricing_route_unavailable")
        route = routes[0]
        pricing = route["pricing"]
        if (
            route["status"] != "live"
            or route["supports_tools"] is not True
            or type(route["context_length"]) is not int
            or route["context_length"] != expected.context_length
            or set(pricing) != {"input", "output"}
            or isinstance(pricing["input"], bool)
            or isinstance(pricing["output"], bool)
            or Decimal(str(pricing["input"])) != Decimal(expected.input_per_million_usd)
            or Decimal(str(pricing["output"]))
            != Decimal(expected.output_per_million_usd)
            or not {"image", "text"} <= set(row["architecture"]["input_modalities"])
            or row["architecture"]["output_modalities"] != ["text"]
        ):
            _deny("hf_pricing_metadata_changed")
        normalized = {
            "model": model,
            "pricing": asdict(expected),
            "status": "live",
            "input_modalities": sorted(row["architecture"]["input_modalities"]),
            "output_modalities": ["text"],
            "supports_tools": True,
        }
        digest = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return HFRouteEvidence(model, expected, digest, time.time())
    except (KeyError, TypeError, ValueError, InvalidOperation):
        _deny("hf_pricing_metadata_invalid")


async def fetch_hf_route(model, *, deadline_at):
    """Share only a running fresh check, with each caller's original deadline."""
    if model not in HF_PRICING:
        _deny("hf_cost_model_not_authorized")
    if not math.isfinite(deadline_at) or deadline_at <= time.time():
        _deny("hf_pricing_deadline_exceeded")
    deadline_at = min(deadline_at, time.time() + 120)
    loop = asyncio.get_running_loop()
    pending = _INFLIGHT.setdefault(loop, {})
    operation = pending.get(model)
    if operation is None or operation.done():
        operation = asyncio.create_task(_wait_for_hf_route(model, deadline_at=deadline_at))
        pending[model] = operation

        def retired(task):
            if pending.get(model) is task:
                pending.pop(model, None)
            if not task.cancelled():
                task.exception()

        operation.add_done_callback(retired)
    try:
        async with asyncio.timeout(max(0.001, deadline_at - time.time())):
            return await asyncio.shield(operation)
    except TimeoutError:
        _deny("hf_pricing_deadline_exceeded")


async def _wait_for_hf_route(model, *, deadline_at):
    """Bound an admission wait without caching or relaxing the route pins."""
    from simready_bench.engine.request_cost import CostDenied

    for attempt in range(5):
        try:
            return await _fetch_hf_route_once(model, deadline_at=deadline_at)
        except CostDenied as error:
            if error.code not in {
                "hf_pricing_route_unavailable",
                "hf_pricing_metadata_unavailable",
            } or attempt == 4:
                raise
            delay = 5 * 2**attempt * random.uniform(0.8, 1.2)
            if time.time() + delay >= deadline_at:
                raise
            await asyncio.sleep(delay)


async def _fetch_hf_route_once(model, *, deadline_at):
    """Validate fresh public metadata without auth, redirects or inference."""
    duration = min(20, deadline_at - time.time())
    if not math.isfinite(duration) or duration <= 0:
        _deny("hf_pricing_deadline_exceeded")
    try:
        async with aiohttp.ClientSession(
            trust_env=False,
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=duration),
        ) as session:
            async with session.get(
                HF_CATALOG_URL,
                allow_redirects=False,
                headers={"Accept-Encoding": "identity"},
            ) as response:
                if response.status != 200:
                    _deny("hf_pricing_metadata_unavailable")
                chunks = []
                size = 0
                async for chunk in response.content.iter_chunked(16384):
                    size += len(chunk)
                    if size > 4 * 1024 * 1024:
                        _deny("hf_pricing_response_limit")
                    chunks.append(chunk)
                catalog = json.loads(b"".join(chunks))
        return validate_catalog_route(catalog, model)
    except (aiohttp.ClientError, TimeoutError, ValueError):
        _deny("hf_pricing_metadata_unavailable")


def hf_upper_cost_micro_usd(model, output_tokens, *, native_output_limits=False):
    if model not in HF_PRICING:
        _deny("hf_cost_model_not_authorized")
    bound = HF_PRICING[model].context_length
    if (
        type(native_output_limits) is not bool
        or type(output_tokens) is not int
        or not 1 <= output_tokens <= bound
    ):
        _deny("output_token_budget_required")
    price = HF_PRICING[model]
    return int(
        (
            Decimal(price.context_length) * Decimal(price.input_per_million_usd)
            + Decimal(output_tokens) * Decimal(price.output_per_million_usd)
        ).to_integral_value(rounding=ROUND_CEILING)
    )
