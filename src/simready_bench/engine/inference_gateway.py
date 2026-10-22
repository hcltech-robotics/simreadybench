"Authenticated model forwarding with input counting and persistent spend limits."

from __future__ import annotations
import asyncio
import base64
from contextlib import closing
from dataclasses import asdict, dataclass, field
import hashlib
import http.server
import inspect
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4
import aiohttp
from aiohttp import web
from aiohttp.web_protocol import RequestHandler
from PIL import Image
from simready_bench.engine.controller import AdmissionError, Controller


@dataclass(frozen=True)
class ProjectBudget:
    project: str
    authorization_id: str
    approved: bool
    upstream_origin: str
    model: str
    expires_at: float
    maximum_requests: int
    maximum_spend_usd: float
    pricing_status: str = "unknown"
    external_spend_cap_verified: bool = False


@dataclass(frozen=True)
class GatewayConfig:
    upstream_base_url: str
    model: str
    upstream_api_key: str = field(repr=False)
    api_style: str = "openai-responses"
    budget: ProjectBudget | None = None
    max_output_tokens: int | None = 4096
    max_request_bytes: int = 64 * 1024 * 1024
    max_concurrency: int = 4
    request_timeout_seconds: float = 1800
    max_connections: int = 16
    header_timeout_seconds: float = 5.0
    listen_backlog: int = 16
    upstream_project_id: str | None = field(default=None, repr=False)
    native_output_cap: bool = False
    native_output_limits: bool = False
    max_input_tokens: int = 64000
    model_spec: dict | None = field(default=None, repr=False)
    strip_native_output_fields: bool = False
    hf_router_relay: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.hf_router_relay is not None:
            from simready_bench.engine.custom_model import validate_hf_router_relay

            validate_hf_router_relay(self.hf_router_relay, base_url=self.upstream_base_url, model=self.model)
        if self.model_spec is not None:
            from simready_bench.engine.custom_model import validate_model_spec

            validate_model_spec(
                self.model_spec,
                model=self.model,
                base_url=self.upstream_base_url,
                api_style=self.api_style,
            )
        if type(self.strip_native_output_fields) is not bool or (
            self.strip_native_output_fields and (not self.native_output_limits)
        ):
            raise ValueError(
                "Client output-field removal requires explicit native output policy"
            )
        if self.upstream_project_id is not None and (
            not isinstance(self.upstream_project_id, str)
            or not re.fullmatch("proj_[A-Za-z0-9_-]{1,128}", self.upstream_project_id)
        ):
            raise ValueError(
                "Fixed OpenAI project identifier required without suffixes or header injection"
            )
        if type(self.native_output_cap) is not bool:
            raise ValueError("native_output_cap must be an explicit trusted boolean")
        from simready_bench.engine.custom_model import configured_facts
        from simready_bench.engine.request_cost import CostDenied

        try:
            provider_facts = configured_facts(self)
        except CostDenied:
            provider_facts = {"maximum_input_tokens": 262144, "output_token_upper_bound": 4096}
        if self.native_output_limits is True:
            from simready_bench.engine.request_cost import CostDenied
            from simready_bench.engine.custom_model import configured_facts

            try:
                provider_input_bound = configured_facts(self)["maximum_input_tokens"]
            except CostDenied as exc:
                raise ValueError(exc.code) from None
            if (
                type(self.max_input_tokens) is not int
                or self.max_input_tokens != provider_input_bound
            ):
                raise ValueError(
                    "Native input admission must use the documented provider maximum"
                )
        elif (
            type(self.max_input_tokens) is not int
            or not 1 <= self.max_input_tokens <= provider_facts["maximum_input_tokens"]
        ):
            raise ValueError(
                "Declared input cap must not exceed the provider maximum"
            )
        parsed = urlsplit(self.upstream_base_url)
        local = parsed.hostname == "localhost"
        try:
            local = local or ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            pass
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (parsed.scheme == "http" and (not local))
        ):
            raise ValueError(
                "Use a fixed HTTPS upstream; plaintext HTTP is restricted to loopback"
            )
        if type(self.native_output_limits) is not bool:
            raise ValueError("native_output_limits must be an explicit trusted boolean")
        if self.native_output_limits:
            if self.max_output_tokens is not None or self.native_output_cap:
                raise ValueError(
                    "Provider-native output requires no benchmark output cap or injection"
                )
            if self.model_spec is None:
                from simready_bench.engine.request_cost import (
                    native_output_reservation_bound,
                    CostDenied,
                )

                try:
                    native_output_reservation_bound(
                        self.model,
                        upstream_origin=self.upstream_origin,
                        api_style=self.api_style,
                    )
                except CostDenied as exc:
                    raise ValueError(exc.code) from None
        elif (
            type(self.max_output_tokens) is not int
            or not 1 <= self.max_output_tokens <= provider_facts["output_token_upper_bound"]
        ):
            raise ValueError("Fixed output cap must be a positive integer within the provider maximum")
        if self.api_style not in ("openai-responses", "openai-completions"):
            raise ValueError(
                "Only the configured OpenAI-compatible transport is supported"
            )
        if (
            not isinstance(self.model, str)
            or not self.model
            or (not isinstance(self.upstream_api_key, str))
            or not self.upstream_api_key
        ):
            raise ValueError(
                "Explicit fixed model and trusted upstream credential required"
            )
        if any(
            (
                type(value) is not int or value <= 0
                for value in (
                    self.max_request_bytes,
                    self.max_concurrency,
                    self.max_connections,
                    self.listen_backlog,
                )
            )
        ):
            raise ValueError("Positive request size and concurrency limits required")
        if (
            type(self.request_timeout_seconds) not in (int, float)
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("Finite positive request timeout required")
        if (
            type(self.header_timeout_seconds) not in (int, float)
            or not math.isfinite(self.header_timeout_seconds)
            or self.header_timeout_seconds <= 0
        ):
            raise ValueError("Finite positive header timeout required")

    @property
    def upstream_origin(self):
        parts = urlsplit(self.upstream_base_url)
        return f"{parts.scheme}://{parts.netloc}"

    @property
    def paths(self):
        if self.api_style == "openai-responses":
            return {
                "/v1/responses": "/responses",
                "/v1/responses/compact": "/responses/compact",
            }
        return {"/v1/chat/completions": "/chat/completions"}


class GatewayDenied(Exception):
    def __init__(self, status, code):
        self.status, self.code = (status, code)
        super().__init__(code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _inline_image_evidence(data):
    """Hash original encoded bytes and decoded first-frame pixels privately.

    RGB/RGBA retain their decoded channel representation; other modes convert
    to RGBA for an actual color hash (palette indices alone are insufficient).
    This observer never rewrites image blocks. Optional native output-cap
    normalization is recorded separately with input and forwarded wire hashes.
    """
    pending = [data]
    images = []
    total_pixels = 0
    deadline = time.monotonic() + 5
    formats = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/gif": "GIF",
        "image/webp": "WEBP",
    }
    while pending:
        if time.monotonic() >= deadline:
            raise GatewayDenied(400, "inline_image_evidence_limit")
        item = pending.pop()
        if isinstance(item, dict):
            kind = item.get("type")
            if kind in ("input_image", "image_url"):
                uri = item["image_url"]
                if isinstance(uri, dict):
                    uri = uri["url"]
                mime = uri.split(";", 1)[0][5:]
                try:
                    raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
                    if len(raw) > 10 * 1024 * 1024 or len(images) >= 128:
                        raise GatewayDenied(400, "inline_image_evidence_limit")
                    with Image.open(
                        io.BytesIO(raw), formats=tuple(formats.values())
                    ) as im:
                        pixels = im.width * im.height
                        if (
                            im.format != formats[mime]
                            or im.width <= 0
                            or im.height <= 0
                        ):
                            raise GatewayDenied(400, "invalid_inline_image")
                        if (
                            pixels > 4096 * 4096
                            or total_pixels + pixels > 64 * 1024 * 1024
                        ):
                            raise GatewayDenied(400, "inline_image_evidence_limit")
                        im.load()
                        decoded = (
                            im if im.mode in ("RGB", "RGBA") else im.convert("RGBA")
                        )
                        images.append(
                            {
                                "mime": mime,
                                "width": im.width,
                                "height": im.height,
                                "mode": decoded.mode,
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "pixels_sha256": hashlib.sha256(
                                    decoded.tobytes()
                                ).hexdigest(),
                            }
                        )
                        total_pixels += pixels
                except GatewayDenied:
                    raise
                except (OSError, ValueError, KeyError, Image.DecompressionBombError):
                    raise GatewayDenied(400, "invalid_inline_image") from None
            pending.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            pending.extend(reversed(item))
    if time.monotonic() >= deadline:
        raise GatewayDenied(400, "inline_image_evidence_limit")
    return {"inline_image_count": len(images), "inline_images": images}


class _Telemetry:
    """Bounded sideband inspection; never alters forwarded response bytes."""

    def __init__(self, streaming):
        self.streaming = streaming
        self.buffer = b""
        self.oversized = False
        self.models = set()
        self.usage = {}
        self.response_ids = set()
        self.finish_reasons = set()
        self.response_statuses = set()
        self.response_identity_overflow = False

    def _bounded_value(self, target, value, *, identity=False):
        if isinstance(value, str) and re.fullmatch("[A-Za-z0-9._:/-]{1,256}", value):
            if value not in target and len(target) >= 16:
                if identity:
                    self.response_identity_overflow = True
                return
            target.add(value)

    def inspect(self, data):
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError):
            return
        if not isinstance(value, dict):
            return
        candidates = [value]
        if isinstance(value.get("response"), dict):
            candidates.append(value["response"])
        for item in candidates:
            self._bounded_value(self.response_ids, item.get("id"), identity=True)
            self._bounded_value(self.response_statuses, item.get("status"))
            incomplete = item.get("incomplete_details")
            if isinstance(incomplete, dict):
                self._bounded_value(self.finish_reasons, incomplete.get("reason"))
            choices = item.get("choices")
            if isinstance(choices, list):
                for choice in choices[:64]:
                    if isinstance(choice, dict):
                        self._bounded_value(
                            self.finish_reasons, choice.get("finish_reason")
                        )
            model = item.get("model")
            if (
                len(self.models) < 16
                and isinstance(model, str)
                and re.fullmatch("[A-Za-z0-9._:/-]{1,256}", model)
            ):
                self.models.add(model)
            usage = item.get("usage")
            if isinstance(usage, dict):
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                ):
                    amount = usage.get(key)
                    if type(amount) is int and amount >= 0:
                        self.usage[key] = amount
                for key in (
                    "input_tokens_details",
                    "output_tokens_details",
                    "prompt_tokens_details",
                    "completion_tokens_details",
                ):
                    detail = usage.get(key)
                    if isinstance(detail, dict):
                        safe = {
                            k: v
                            for k, v in detail.items()
                            if isinstance(k, str)
                            and re.fullmatch("[A-Za-z_]{1,64}", k)
                            and (type(v) is int)
                            and (v >= 0)
                        }
                        if safe:
                            self.usage[key] = safe

    def feed(self, data):
        if self.oversized:
            return
        self.buffer += data
        if len(self.buffer) > 1024 * 1024:
            self.buffer = b""
            self.oversized = True
            return
        if self.streaming:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line.startswith(b"data:"):
                    self.inspect(line[5:].strip())

    def finish(self):
        if not self.streaming and (not self.oversized):
            self.inspect(self.buffer)


_BOUNDED_RUNNER = web.AppKey("simready_bounded_gateway_runner")
_SUPPORTED_AIOHTTP = "3.14.5"


class _RejectedConnection(asyncio.Protocol):
    def connection_made(self, transport):
        transport.write(
            b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
        )
        transport.close()


class _BoundedRequestHandler(RequestHandler):
    def __init__(self, manager, **kwargs):
        super().__init__(manager, **kwargs)
        self._initial_header_timer = self._loop.call_later(
            manager.header_timeout_seconds, self._expire_initial_headers
        )

    def _cancel_header_timer(self):
        if self._initial_header_timer is not None:
            self._initial_header_timer.cancel()
            self._initial_header_timer = None

    def _expire_initial_headers(self):
        self._initial_header_timer = None
        had_transport = self.transport is not None
        self.force_close()
        if not had_transport and self._manager is not None:
            self._manager.release_slot(self)

    def connection_made(self, transport):
        if self._force_close:
            transport.close()
            self._manager.release_slot(self)
            return
        try:
            super().connection_made(transport)
        except BaseException:
            self._cancel_header_timer()
            self._manager.release_slot(self)
            transport.close()
            raise

    def data_received(self, data):
        super().data_received(data)
        if self._request_count:
            self._cancel_header_timer()

    def connection_lost(self, exc):
        self._cancel_header_timer()
        manager = self._manager
        try:
            super().connection_lost(exc)
        finally:
            if manager is not None:
                manager.release_slot(self)


class _BoundedServer(web.Server):
    def __init__(self, *args, connection_limit, header_timeout_seconds, **kwargs):
        super().__init__(*args, **kwargs)
        self.connection_limit = connection_limit
        self.header_timeout_seconds = header_timeout_seconds
        self._slots = set()

    @property
    def connection_slots_in_use(self):
        return len(self._slots)

    def release_slot(self, handler):
        self._slots.discard(handler)

    def __call__(self):
        if len(self._slots) >= self.connection_limit:
            return _RejectedConnection()
        handler = _BoundedRequestHandler(self, loop=self._loop, **self._kwargs)
        self._slots.add(handler)
        return handler


class _BoundedUnixSite(web.UnixSite):
    async def start(self):
        await super().start()
        os.chmod(self._path, 384)


class _BoundedAppRunner(web.AppRunner):
    def __init__(self, gateway, *, shutdown_timeout=1):
        if aiohttp.__version__ != _SUPPORTED_AIOHTTP:
            raise RuntimeError(
                "The bounded gateway protocol requires pinned aiohttp "
                + _SUPPORTED_AIOHTTP
            )
        self._gateway_config = gateway.config
        self.listen_backlog = gateway.config.listen_backlog
        super().__init__(
            gateway.application(),
            access_log=None,
            shutdown_timeout=shutdown_timeout,
            handler_cancellation=True,
            keepalive_timeout=gateway.config.header_timeout_seconds,
            lingering_time=0,
            max_headers=64,
            max_line_size=8190,
            max_field_size=8190,
            read_bufsize=65536,
            auto_decompress=False,
        )

    async def _make_server(self):
        self._app[_BOUNDED_RUNNER] = self
        original = await super()._make_server()
        return _BoundedServer(
            original.request_handler,
            request_factory=original.request_factory,
            loop=original._loop,
            handler_cancellation=True,
            connection_limit=self._gateway_config.max_connections,
            header_timeout_seconds=self._gateway_config.header_timeout_seconds,
            **original._kwargs,
        )

    def tcp_site(self, host="127.0.0.1", port=0):
        """Normal aiohttp TCP site with the frozen backlog; no sockets yet."""
        return web.TCPSite(self, host, port, backlog=self.listen_backlog)

    def unix_site(self, path):
        """Private Unix site with the same parser slots/header/backlog bounds."""
        path = Path(path).absolute()
        if (
            path.parent.resolve() != path.parent
            or path.exists()
            or path.parent.stat().st_uid != os.getuid()
            or path.parent.stat().st_mode & 18
        ):
            raise ValueError("Fresh Unix socket path in a trusted directory required")
        if len(str(path).encode()) > 100:
            raise ValueError("Unix socket path exceeds portable bound")
        return _BoundedUnixSite(self, path, backlog=self.listen_backlog)

    def _reg_site(self, site):
        if (
            not isinstance(site, (web.TCPSite, _BoundedUnixSite))
            or site._backlog != self.listen_backlog
        ):
            raise RuntimeError("Use the bounded runner TCP or Unix gateway listener")
        super()._reg_site(site)


class InferenceGateway:
    def __init__(
        self, controller: Controller, config: GatewayConfig, *, on_stop_requested=None
    ):
        self.controller = controller
        self.config = config
        self.on_stop_requested = on_stop_requested
        self.db_path = controller.state_dir / "inference-gateway.sqlite3"
        self._tokens = {}
        self._token_lock = threading.Lock()
        self._session = None
        self._semaphore = None
        self._admission_lock = None
        self._stop_notifications = {}
        self._provider_relay_server = None
        self._provider_relay_thread = None
        self._provider_relay_base_url = None
        self._provider_relay_marker = None
        with closing(self._db()) as db:
            db.executescript(
                "\n                CREATE TABLE IF NOT EXISTS budgets (\n                    authorization_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,\n                    requests_used INTEGER NOT NULL);\n                CREATE TABLE IF NOT EXISTS requests (\n                    request_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL,\n                    controller_job_id TEXT NOT NULL, record TEXT NOT NULL);\n                CREATE TABLE IF NOT EXISTS stop_requests (\n                    episode_id TEXT NOT NULL, reason TEXT NOT NULL,\n                    callback_status TEXT NOT NULL DEFAULT 'pending',\n                    PRIMARY KEY(episode_id,reason));\n            "
            )
            if "callback_status" not in {
                row["name"] for row in db.execute("PRAGMA table_info(stop_requests)")
            }:
                db.execute(
                    "ALTER TABLE stop_requests ADD COLUMN callback_status TEXT NOT NULL DEFAULT 'pending'"
                )
        self.db_path.chmod(384)

    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    @property
    def _request_time_limit(self):
        return self.config.request_timeout_seconds

    def _budget(self):
        from simready_bench.engine.request_cost import supports_application_spend_cap

        value = self.config.budget
        if (
            not isinstance(value, ProjectBudget)
            or value.project != "simready-bench"
            or value.approved is not True
            or (not value.authorization_id)
            or (value.upstream_origin != self.config.upstream_origin)
            or (value.model != self.config.model)
            or (type(value.expires_at) not in (int, float))
            or (not math.isfinite(value.expires_at))
            or (value.expires_at <= time.time())
            or (type(value.maximum_requests) is not int)
            or (value.maximum_requests <= 0)
            or (type(value.maximum_spend_usd) not in (int, float))
            or (not math.isfinite(value.maximum_spend_usd))
            or (value.maximum_spend_usd <= 0)
            or (value.pricing_status not in ("known", "unknown"))
            or (
                value.external_spend_cap_verified is not True
                and (not supports_application_spend_cap(self.config))
            )
        ):
            raise GatewayDenied(503, "project_budget_unavailable")
        return value

    def issue_episode_token(self, episode_id):
        """Trusted-only API. Tokens are ephemeral; only their hashes are retained."""
        episode = self.controller.get_episode(episode_id)
        if episode.status != "active" or time.time() >= episode.deadline_at:
            raise AdmissionError("episode_not_active")
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._token_lock:
            self._tokens[digest] = (episode_id, episode.deadline_at)
        return token

    def revoke_episode(self, episode_id):
        with self._token_lock:
            self._tokens = {
                key: value
                for key, value in self._tokens.items()
                if value[0] != episode_id
            }

    def _authenticate(self, request):
        authorization = request.headers.getall("Authorization", [])
        if len(authorization) != 1 or not authorization[0].startswith("Bearer "):
            raise GatewayDenied(401, "invalid_episode_token")
        token = authorization[0][7:]
        if not 16 <= len(token) <= 256:
            raise GatewayDenied(401, "invalid_episode_token")
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._token_lock:
            selected = self._tokens.get(digest)
        if selected is None or time.time() >= selected[1]:
            raise GatewayDenied(401, "invalid_episode_token")
        return selected[0]

    async def reserve_request_cost(
        self, payload, request_id, purpose, deadline_at, *, episode_id=None
    ):
        """Actual fixed provider count plus shared durable monetary admission.

        Judge and actor use the same project ledger. A reservation is an upper
        bound, never an observed charge; failed or interrupted attempts retain it.
        No provider counter is called until _budget authorizes this exact origin.
        """
        from simready_bench.engine.request_cost import (
            CostDenied,
            CostLedger,
            count_input_tokens,
            supports_hf_application_spend_cap,
            verify_hf_context_bound,
        )

        selected = self._budget()
        self._validate_payload(_json(payload).encode())
        if purpose not in ("actor", "judge"):
            raise GatewayDenied(400, "invalid_cost_purpose")
        if (
            type(deadline_at) not in (int, float)
            or not math.isfinite(deadline_at)
            or deadline_at <= time.time()
        ):
            raise GatewayDenied(504, "input_counter_deadline_exceeded")
        try:
            ledger = CostLedger(self.controller.state_dir)
            if self.config.model_spec is not None:
                from simready_bench.engine.custom_model import reserve_custom_cost, verify_custom_hf_route

                route = await verify_custom_hf_route(self.config, deadline_at=min(deadline_at, selected.expires_at))
                reservation = await asyncio.to_thread(
                    reserve_custom_cost,
                    self.config,
                    payload,
                    ledger,
                    request_id,
                    purpose,
                    route=route,
                )
            else:
                counter = (
                    verify_hf_context_bound
                    if supports_hf_application_spend_cap(self.config)
                    else count_input_tokens
                )
                evidence = await counter(
                    self.config,
                    payload,
                    maximum_input_tokens=self.config.max_input_tokens,
                    deadline_at=min(deadline_at, selected.expires_at),
                )
                reservation = await asyncio.to_thread(
                    ledger.reserve,
                    budget=selected,
                    payload=payload,
                    evidence=evidence,
                    request_id=request_id,
                    purpose=purpose,
                    upstream_project_id=self.config.upstream_project_id,
                    maximum_input_tokens=self.config.max_input_tokens,
                    native_output_limits=self.config.native_output_limits,
                )
            return reservation
        except CostDenied as exc:
            status = (
                400
                if exc.code
                in {
                    "input_token_budget_exceeded",
                    "output_token_budget_required",
                    "cost_model_not_authorized",
                    "verified_input_count_requires_responses",
                }
                else 429
                if exc.code == "project_spend_budget_exhausted"
                else 503
            )
            raise GatewayDenied(status, exc.code) from None

    def _validate_payload(self, raw):
        def reject_constant(_):
            raise ValueError("nonfinite JSON")

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        try:
            data = json.loads(
                raw, parse_constant=reject_constant, object_pairs_hook=unique_object
            )
        except (ValueError, UnicodeError, RecursionError):
            raise GatewayDenied(400, "invalid_json") from None
        if not isinstance(data, dict) or data.get("model") != self.config.model:
            raise GatewayDenied(400, "model_not_authorized")
        if self.config.strip_native_output_fields:
            for key in ("max_output_tokens", "max_tokens", "max_completion_tokens"):
                data.pop(key, None)
        forbidden = {
            "base_url",
            "upstream_url",
            "api_key",
            "authorization",
            "headers",
            "provider",
            "endpoint",
            "url",
            "organization",
            "project",
            "episode_id",
            "extra_body",
            "extra_headers",
            "request_options",
            "previous_response_id",
            "conversation",
            "prompt",
        }
        if (
            forbidden.intersection(data)
            or data.get("store") is True
            or data.get("background") is True
        ):
            raise GatewayDenied(400, "request_option_not_authorized")
        common = {
            "model",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "temperature",
            "top_p",
            "stream",
            "stream_options",
            "store",
            "metadata",
            "user",
            "safety_identifier",
            "prompt_cache_key",
        }
        allowed = (
            common
            | {
                "input",
                "instructions",
                "text",
                "reasoning",
                "max_output_tokens",
                "background",
                "truncation",
                "include",
            }
            if self.config.api_style == "openai-responses"
            else common
            | {
                "messages",
                "response_format",
                "reasoning_effort",
                "thinking",
                "max_tokens"
                if self.config.hf_router_relay is not None
                or self.config.upstream_base_url == "https://router.huggingface.co/v1"
                else "max_completion_tokens",
                "stop",
                "seed",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "logprobs",
                "top_logprobs",
            }
        )
        if set(data) - allowed:
            raise GatewayDenied(400, "request_option_not_authorized")
        if "thinking" in data:
            thinking = data["thinking"]
            if (
                not isinstance(thinking, dict)
                or set(thinking) - {"type", "clear_thinking"}
                or thinking.get("type") not in ("enabled", "disabled")
                or ("clear_thinking" in thinking and type(thinking["clear_thinking"]) is not bool)
            ):
                raise GatewayDenied(400, "invalid_thinking_setting")
        affinity = data.get("prompt_cache_key")
        if "prompt_cache_key" in data and (
            not isinstance(affinity, str)
            or not re.fullmatch("[A-Za-z0-9_.:-]{1,128}", affinity)
        ):
            raise GatewayDenied(400, "invalid_cache_affinity")
        output_field = (
            "max_output_tokens"
            if self.config.api_style == "openai-responses"
            else "max_tokens"
            if self.config.hf_router_relay is not None
            or self.config.upstream_base_url == "https://router.huggingface.co/v1"
            else "max_completion_tokens"
        )
        output = data.get(output_field)
        if self.config.native_output_limits:
            if any(
                (
                    k in data
                    for k in (
                        "max_output_tokens",
                        "max_tokens",
                        "max_completion_tokens",
                    )
                )
            ):
                raise GatewayDenied(400, "native_output_limit_field_forbidden")
        elif output_field not in data and self.config.native_output_cap:
            data[output_field] = self.config.max_output_tokens
            output = data[output_field]
        if not self.config.native_output_limits and (
            type(output) is not int or not 1 <= output <= self.config.max_output_tokens
        ):
            raise GatewayDenied(400, "output_token_limit_required")
        if "stream" in data and type(data["stream"]) is not bool:
            raise GatewayDenied(400, "invalid_stream_setting")
        tools = data.get("tools", [])
        if not isinstance(tools, list) or any(
            (
                not isinstance(tool, dict)
                or tool.get("type") not in ("function", "custom")
                for tool in tools
            )
        ):
            raise GatewayDenied(400, "provider_tool_not_authorized")
        pending, seen = ([(data, 0)], 0)
        while pending:
            item, depth = pending.pop()
            seen += 1
            if seen > 100000 or depth > 64:
                raise GatewayDenied(400, "request_structure_limit")
            if isinstance(item, dict):
                kind = item.get("type")
                if {"prompt_cache_breakpoint", "cache_control"}.intersection(item):
                    raise GatewayDenied(400, "request_option_not_authorized")
                if kind in (
                    "input_file",
                    "file",
                    "file_url",
                    "item_reference",
                    "input_audio",
                    "audio_url",
                    "input_video",
                    "video_url",
                ):
                    raise GatewayDenied(400, "remote_input_not_authorized")
                if kind in ("input_image", "image_url"):
                    if "file_id" in item or "file_url" in item:
                        raise GatewayDenied(400, "remote_input_not_authorized")
                    value = item.get("image_url")
                    if kind == "image_url" and isinstance(value, dict):
                        value = value.get("url")
                    if not isinstance(value, str) or not re.match(
                        "^data:image/(png|jpeg|gif|webp);base64,", value
                    ):
                        raise GatewayDenied(400, "inline_image_required")
                    try:
                        base64.b64decode(value.split(",", 1)[1], validate=True)
                    except ValueError:
                        raise GatewayDenied(400, "invalid_inline_image") from None
                pending.extend(((value, depth + 1) for value in item.values()))
            elif isinstance(item, list):
                pending.extend(((value, depth + 1) for value in item))
        return data

    def _check_request_capacity(self, episode_id):
        """Reject exhausted requests before counter/route calls or money holds.

        The admission lock serializes this check, costing and final admission
        within this gateway. _admit remains the persistent atomic dispatch guard
        across gateway instances; failed cost admission still consumes no call.
        """
        selected = self._budget()
        with closing(self._db()) as db:
            row = db.execute(
                "SELECT requests_used FROM budgets WHERE authorization_id=?",
                (selected.authorization_id,),
            ).fetchone()
        if row is not None and row["requests_used"] >= selected.maximum_requests:
            raise GatewayDenied(429, "project_request_budget_exhausted")
        episode = self.controller.get_episode(episode_id)
        if episode.inference_calls_used >= episode.inference_limit:
            raise AdmissionError("inference_budget_exhausted")

    def _admit(self, episode_id, record):
        selected = self._budget()
        identity = asdict(selected)
        if self.config.upstream_project_id is not None:
            identity = {
                **identity,
                "upstream_project_sha256": hashlib.sha256(
                    self.config.upstream_project_id.encode()
                ).hexdigest(),
            }
        fingerprint = hashlib.sha256(_json(identity).encode()).hexdigest()
        db = self._db()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM budgets WHERE authorization_id=?",
                (selected.authorization_id,),
            ).fetchone()
            if row and row["fingerprint"] != fingerprint:
                raise GatewayDenied(503, "budget_authorization_conflict")
            used = row["requests_used"] if row else 0
            if used >= selected.maximum_requests:
                raise GatewayDenied(429, "project_request_budget_exhausted")
            job = self.controller.reserve_inference(
                episode_id, record["gateway_request_id"]
            )
            db.execute(
                "INSERT INTO budgets VALUES (?,?,?) ON CONFLICT(authorization_id) DO UPDATE SET requests_used=excluded.requests_used",
                (selected.authorization_id, fingerprint, used + 1),
            )
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,?)",
                (record["gateway_request_id"], episode_id, job.job_id, _json(record)),
            )
            db.commit()
            return (job, used + 1 >= selected.maximum_requests)
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _save(self, job, record):
        with closing(self._db()) as db:
            db.execute(
                "UPDATE requests SET record=? WHERE request_id=?",
                (_json(record), record["gateway_request_id"]),
            )
        try:
            self.controller.complete_job(job.job_id, record)
        except AdmissionError as exc:
            if exc.code != "job_cancelled":
                raise

    def records(self, episode_id=None):
        """Trusted observability API, never an actor HTTP route."""
        with closing(self._db()) as db:
            rows = db.execute(
                "SELECT record FROM requests"
                + (" WHERE episode_id=?" if episode_id is not None else "")
                + " ORDER BY rowid",
                (episode_id,) if episode_id is not None else (),
            ).fetchall()
            return tuple((json.loads(row["record"]) for row in rows))

    async def _notify(self, episode_id, reason):
        key = (episode_id, reason)
        delivery = self._stop_notifications.get(key)
        if delivery is None:
            delivery = asyncio.create_task(self._deliver_stop(episode_id, reason))
            self._stop_notifications[key] = delivery
        # The actor's connection disappears when this callback stops it. Keep
        # delivery and its durable outcome independent of that request handler.
        await asyncio.shield(delivery)

    async def _deliver_stop(self, episode_id, reason):
        def reserve():
            with closing(self._db()) as db:
                return (
                    db.execute(
                        "INSERT OR IGNORE INTO stop_requests(episode_id,reason) VALUES (?,?)",
                        (episode_id, reason),
                    ).rowcount
                    == 1
                )

        if await asyncio.to_thread(reserve) and self.on_stop_requested is not None:
            status = "delivered"
            try:
                result = self.on_stop_requested(episode_id, reason)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                status = "failed"

            def save_status():
                with closing(self._db()) as db:
                    db.execute(
                        "UPDATE stop_requests SET callback_status=? WHERE episode_id=? AND reason=?",
                        (status, episode_id, reason),
                    )

            await asyncio.to_thread(save_status)

    def stop_requests(self):
        """Trusted observer API. Delivery to a callback never asserts termination."""
        with closing(self._db()) as db:
            return tuple(
                (
                    dict(row)
                    for row in db.execute(
                        "SELECT episode_id,reason,callback_status FROM stop_requests ORDER BY rowid"
                    )
                )
            )

    async def _watch_episode(self, episode_id, request, task, closure):
        while True:
            await asyncio.sleep(0.05)
            try:
                episode = await asyncio.to_thread(
                    self.controller.get_episode, episode_id
                )
            except Exception:
                closure.append("controller_unavailable")
            else:
                if episode.status != "active":
                    closure.append("episode_admission_closed")
                elif time.time() >= episode.deadline_at:
                    closure.append("inference_deadline_exceeded")
                elif request.transport is None or request.transport.is_closing():
                    closure.append("client_disconnected")
            if closure:
                task.cancel()
                return

    def runner(self, *, shutdown_timeout=1):
        """Supported serving API: await runner.setup(), then runner.tcp_site().start().

        The pinned aiohttp protocol enforces incoming connection and initial
        header bounds. A plain AppRunner cannot serve this application.
        """
        return _BoundedAppRunner(self, shutdown_timeout=shutdown_timeout)

    def application(self):
        """Handler bundle only. Use runner() to serve with transport bounds."""
        app = web.Application(client_max_size=self.config.max_request_bytes)

        async def start(_):
            runner = app.get(_BOUNDED_RUNNER)
            if not isinstance(runner, _BoundedAppRunner) or runner.app is not app:
                raise RuntimeError("Serve the gateway through its bounded runner()")
            if urlsplit(self.config.upstream_base_url).scheme == "https":
                from simready_bench.provider_relay import Relay, RelayState

                budget = self._budget()
                marker = secrets.token_urlsafe(32)
                directory = (
                    self.controller.state_dir
                    / "provider-relay"
                    / hashlib.sha256(budget.authorization_id.encode()).hexdigest()
                )
                state = RelayState(
                    {
                        "model": self.config.model,
                        "marker": marker,
                        "api_key": self.config.upstream_api_key,
                        "base_url": self.config.upstream_base_url,
                        "project_id": self.config.upstream_project_id,
                        "maximum_requests": budget.maximum_requests,
                        "maximum_condition_requests": budget.maximum_requests,
                        "expires_at": budget.expires_at,
                        "native_output_only": self.config.native_output_limits,
                        "maximum_request_bytes": min(
                            self.config.max_request_bytes, 64 * 1024**2
                        ),
                    },
                    directory=directory,
                )
                server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Relay)
                server.daemon_threads = True
                server.state = state
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                self._provider_relay_server = server
                self._provider_relay_thread = thread
                self._provider_relay_marker = marker
                self._provider_relay_base_url = (
                    "http://127.0.0.1:" + str(server.server_port) + "/v1"
                )
            self._session = aiohttp.ClientSession(
                trust_env=False,
                auto_decompress=False,
                cookie_jar=aiohttp.DummyCookieJar(),
                skip_auto_headers={"User-Agent", "Accept-Encoding", "Content-Type"},
            )
            self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
            self._admission_lock = asyncio.Lock()

        async def close(_):
            if self._stop_notifications:
                await asyncio.gather(
                    *self._stop_notifications.values(), return_exceptions=True
                )
            if self._session is not None:
                await self._session.close()
                self._session = None
            if self._provider_relay_server is not None:
                await asyncio.to_thread(self._provider_relay_server.shutdown)
                self._provider_relay_server.server_close()
                await asyncio.to_thread(self._provider_relay_thread.join, 2)
                self._provider_relay_server = None
                self._provider_relay_thread = None
                self._provider_relay_base_url = None
                self._provider_relay_marker = None

        app.on_startup.append(start)
        app.on_cleanup.append(close)
        app.router.add_route("*", "/{path:.*}", self.handle)
        return app

    async def _reconcile_usage(self, record):
        if record.get("usage_reconciliation_attempted") or not record.get(
            "reserved_upper_cost_micro_usd"
        ):
            return
        record["usage_reconciliation_attempted"] = True
        if record.get("status") != "completed" or record.get("upstream_status") != 200:
            return
        from simready_bench.engine.request_cost import (
            CostLedger,
            make_usage_settlement_evidence,
        )

        try:
            evidence = make_usage_settlement_evidence(record)
            result = await asyncio.to_thread(
                CostLedger(self.controller.state_dir).settle,
                record["gateway_request_id"],
                evidence=evidence,
                cost_payload_sha256=record.get("cost_payload_sha256"),
            )
            record["project_usage_settlement"] = {
                k: v for k, v in result.items() if k != "proof"
            }
        except Exception as exc:
            record["usage_settlement_error"] = {
                "error_type": type(exc).__name__,
                "code": getattr(exc, "code", "settlement_unavailable"),
            }

    async def handle(self, request):
        job, response, record, episode_id = (None, None, None, None)
        denial = None
        watchdog, closure = (None, [])
        telemetry, response_hash, response_bytes = (None, None, 0)
        started = time.monotonic()
        try:
            episode_id = self._authenticate(request)
            if (
                request.method != "POST"
                or request.path not in self.config.paths
                or request.query_string
            ):
                raise GatewayDenied(400, "endpoint_not_authorized")
            if any(
                (
                    key.lower()
                    in {
                        "x-api-key",
                        "api-key",
                        "proxy-authorization",
                        "openai-organization",
                        "openai-project",
                        "cookie",
                    }
                    or key.lower().startswith("x-simready-")
                    for key in request.headers
                )
            ):
                raise GatewayDenied(400, "credential_header_not_authorized")
            if request.headers.get("Content-Encoding", "identity") != "identity":
                raise GatewayDenied(400, "encoded_request_not_authorized")
            if request.content_type != "application/json":
                raise GatewayDenied(400, "json_content_type_required")
            if request.content_length is None:
                raise GatewayDenied(411, "content_length_required")
            if request.content_length > self.config.max_request_bytes:
                raise GatewayDenied(413, "request_too_large")
            self._budget()
            if request.path == "/v1/responses/compact":
                raise GatewayDenied(400, "compaction_output_bound_unavailable")
            if self._semaphore.locked():
                raise GatewayDenied(429, "gateway_busy")
            async with self._semaphore:
                async with asyncio.timeout(min(30, self._request_time_limit)):
                    raw = await request.read()
                input_wire_hash = hashlib.sha256(raw).hexdigest()
                data = self._validate_payload(raw)
                native_normalization = {}
                if (
                    self.config.native_output_cap
                    or self.config.strip_native_output_fields
                ):
                    normalized = _json(data).encode()
                    native_normalization = {
                        "input_wire_sha256": input_wire_hash,
                        "forwarded_payload_sha256": hashlib.sha256(
                            normalized
                        ).hexdigest(),
                        "native_output_cap_enabled": True,
                        "payload_normalized": normalized != raw,
                        "image_values_preserved": True,
                    }
                    original = json.loads(raw)
                    inserted = [
                        k
                        for k in (
                            "max_output_tokens",
                            "max_tokens",
                            "max_completion_tokens",
                        )
                        if k not in original and k in data
                    ]
                    native_normalization["injected_output_token_field"] = (
                        inserted[0] if inserted else None
                    )
                    if self.config.strip_native_output_fields:
                        native_normalization.update(
                            native_output_cap_enabled=False,
                            removed_output_token_fields=[
                                k
                                for k in (
                                    "max_output_tokens",
                                    "max_tokens",
                                    "max_completion_tokens",
                                )
                                if k in original
                            ],
                        )
                    raw = normalized
                    if len(raw) > self.config.max_request_bytes:
                        raise GatewayDenied(413, "request_too_large")
                image_evidence = await asyncio.to_thread(_inline_image_evidence, data)
                record = {
                    "gateway_request_id": uuid4().hex,
                    "episode_id": episode_id,
                    "model": self.config.model,
                    "api_path": request.path,
                    "request_sha256": hashlib.sha256(raw).hexdigest(),
                    "request_bytes": len(raw),
                    "stream": data.get("stream", False),
                    "maximum_output_tokens": self.config.max_output_tokens,
                    "requested_output_token_cap": self.config.max_output_tokens,
                    "effective_output_token_cap": next((data[k] for k in ("max_output_tokens", "max_tokens", "max_completion_tokens") if k in data), None),
                    "status": "admitted",
                    "cost_usd": None,
                    "cost_status": "unknown",
                }
                if self.config.native_output_limits:
                    from simready_bench.engine.custom_model import configured_facts

                    record.update(configured_facts(self.config))
                record.update(image_evidence)
                record.update(native_normalization)
                async with self._admission_lock:
                    await asyncio.to_thread(self._check_request_capacity, episode_id)
                    episode = await asyncio.to_thread(
                        self.controller.get_episode, episode_id
                    )
                    deadline = min(
                        episode.deadline_at or time.time(),
                        time.time()
                        + self.config.request_timeout_seconds
                        - (time.monotonic() - started),
                    )
                    cost = await self.reserve_request_cost(
                        data,
                        record["gateway_request_id"],
                        "actor",
                        deadline,
                        episode_id=episode_id,
                    )
                    cost["cost_payload_sha256"] = cost.pop("request_sha256")
                    record.update(cost)
                    job, _ = await asyncio.to_thread(self._admit, episode_id, record)
                await asyncio.to_thread(self.controller.mark_job_running, job.job_id)
                watchdog = asyncio.create_task(
                    self._watch_episode(
                        episode_id, request, asyncio.current_task(), closure
                    )
                )
                remaining = min(
                    job.deadline_at - time.time(),
                    self._request_time_limit - (time.monotonic() - started),
                    self._budget().expires_at - time.time(),
                )
                if remaining <= 0:
                    raise GatewayDenied(504, "inference_deadline_exceeded")
                telemetry = _Telemetry(data.get("stream", False))
                response_hash, response_bytes = (hashlib.sha256(), 0)
                headers = {
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                    "Accept": "text/event-stream"
                    if data.get("stream")
                    else "application/json",
                }
                if self.config.upstream_api_key:
                    headers["Authorization"] = "Bearer " + self.config.upstream_api_key
                if self.config.upstream_project_id is not None:
                    headers["OpenAI-Project"] = self.config.upstream_project_id
                forwarding_base_url = (
                    self._provider_relay_base_url or self.config.upstream_base_url
                )
                private_transport = forwarding_base_url.startswith(
                    ("http://127.0.0.1:", "http://localhost:")
                )
                if self._provider_relay_marker is not None:
                    headers["Authorization"] = "Bearer " + self._provider_relay_marker
                    headers.pop("OpenAI-Project", None)
                if private_transport:
                    headers.update(
                        {
                            "X-SimReady-Condition": self._budget().authorization_id,
                            "X-SimReady-Request-Limit": str(
                                min(
                                    self._budget().maximum_requests,
                                    episode.inference_limit,
                                )
                            ),
                            "X-SimReady-Deadline": str(
                                min(
                                    job.deadline_at,
                                    self._budget().expires_at,
                                    time.time() + remaining,
                                )
                            ),
                        }
                    )
                url = forwarding_base_url.rstrip("/") + self.config.paths[request.path]
                async with asyncio.timeout(remaining):
                    async with self._session.post(
                        url,
                        data=raw,
                        headers=headers,
                        allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(
                            total=remaining, connect=min(10, remaining)
                        ),
                    ) as upstream:
                        record["observed_upstream_status"] = upstream.status
                        for header, field in (
                            ("X-SimReady-Upstream-Attempts", "upstream_attempts"),
                            (
                                "X-SimReady-Scope-Requests",
                                "upstream_condition_requests",
                            ),
                        ):
                            value = upstream.headers.get(header)
                            if (
                                value is not None
                                and len(value) <= 10
                                and value.isascii()
                                and value.isdecimal()
                                and private_transport
                            ):
                                amount = int(value)
                                bound = (
                                    8
                                    if field == "upstream_attempts"
                                    else self._budget().maximum_requests
                                )
                                if 0 <= amount <= bound:
                                    record[field] = amount
                        if 300 <= upstream.status < 400:
                            raise GatewayDenied(502, "upstream_redirect_rejected")
                        selected_headers = {
                            key: value
                            for key, value in upstream.headers.items()
                            if key.lower()
                            in (
                                "content-type",
                                "content-encoding",
                                "cache-control",
                                "retry-after",
                            )
                        }
                        response = web.StreamResponse(
                            status=upstream.status, headers=selected_headers
                        )
                        await response.prepare(request)
                        async for chunk in upstream.content.iter_chunked(65536):
                            response_hash.update(chunk)
                            response_bytes += len(chunk)
                            telemetry.feed(chunk)
                            await response.write(chunk)
                        telemetry.finish()
                        if upstream.status == 200 and data.get("stream"):
                            terminal = (
                                bool(telemetry.finish_reasons)
                                if self.config.api_style == "openai-completions"
                                else bool(
                                    telemetry.response_statuses.intersection(
                                        {"completed", "incomplete", "failed", "cancelled"}
                                    )
                                )
                            )
                            if not terminal:
                                # HTTP EOF can hide a relay/provider disconnect.
                                # Preserve the admission and partial evidence,
                                # but never send a successful downstream EOF.
                                raise GatewayDenied(502, "upstream_stream_incomplete")
                        record.update(
                            status="completed",
                            upstream_status=upstream.status,
                            response_sha256=response_hash.hexdigest(),
                            response_bytes=response_bytes,
                            provider_reported_models=sorted(telemetry.models),
                            usage=telemetry.usage,
                            provider_response_ids=sorted(telemetry.response_ids),
                            provider_finish_reasons=sorted(telemetry.finish_reasons),
                            provider_response_statuses=sorted(
                                telemetry.response_statuses
                            ),
                            provider_response_identity_overflow=telemetry.response_identity_overflow
                            or telemetry.oversized,
                        )
                        await self._reconcile_usage(record)
                        await response.write_eof()
                return response
        except AdmissionError as exc:
            if exc.code == "inference_budget_exhausted":
                await self._notify(episode_id, exc.code)
                denial = GatewayDenied(429, exc.code)
            else:
                denial = GatewayDenied(403, "episode_admission_closed")
        except GatewayDenied as exc:
            denial = exc
            if episode_id is not None and exc.code in (
                "project_request_budget_exhausted",
                "project_spend_budget_exhausted",
            ):
                await self._notify(episode_id, exc.code)
        except (asyncio.TimeoutError, TimeoutError):
            denial = GatewayDenied(504, "inference_deadline_exceeded")
        except (aiohttp.ClientError, ConnectionError):
            denial = GatewayDenied(502, "upstream_transport_error")
        except asyncio.CancelledError:
            if record is not None:
                record["status"] = "cancelled"
            if not closure:
                raise
            asyncio.current_task().uncancel()
            status = 503 if closure[0] == "controller_unavailable" else 403
            if closure[0] == "inference_deadline_exceeded":
                status = 504
            denial = GatewayDenied(status, closure[0])
        except Exception:
            denial = GatewayDenied(500, "gateway_internal_error")
        finally:
            if watchdog is not None:
                watchdog.cancel()
                await asyncio.gather(watchdog, return_exceptions=True)
            if (
                job is not None
                and record is not None
                and (record["status"] != "completed")
                and (telemetry is not None)
            ):
                record.update(
                    partial_response_sha256=response_hash.hexdigest(),
                    partial_response_bytes=response_bytes,
                    observed_provider_reported_models=sorted(telemetry.models),
                    observed_provider_response_ids=sorted(telemetry.response_ids),
                    observed_provider_finish_reasons=sorted(telemetry.finish_reasons),
                    observed_provider_response_statuses=sorted(
                        telemetry.response_statuses
                    ),
                    observed_provider_response_identity_overflow=telemetry.response_identity_overflow
                    or telemetry.oversized,
                    partial_usage_terminal=False,
                )
                if telemetry.usage:
                    record["partial_usage"] = telemetry.usage
            if record is not None and record.get("shared_asset_reservation"):
                await self._reconcile_usage(record)
            if job is not None and record is not None:
                if record["status"] == "admitted":
                    record["status"] = "transport_failed"
                record["elapsed_seconds"] = time.monotonic() - started
                if denial is not None:
                    record["error_code"] = denial.code
                await asyncio.to_thread(self._save, job, record)
        if response is not None and response.prepared:
            response.force_close()
            if request.transport is not None:
                request.transport.close()
            return response
        return web.json_response(
            {"error": {"code": denial.code}},
            status=denial.status,
            headers={"Cache-Control": "no-store"},
        )
