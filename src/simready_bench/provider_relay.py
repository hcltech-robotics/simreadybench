"""Private loopback transport with bounded recovery from provider overload."""

from __future__ import annotations

import hashlib
import hmac
import http.server
import json
import math
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from email.utils import parsedate_to_datetime


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Model provider redirects are not allowed")


class RelayAllowanceError(ValueError):
    """A relay authority limit refused an upstream attempt."""


RETRYABLE_STATUS = {429, 502, 503, 504}
PROBE_BYTES = 64 * 1024


def _overload(error):
    if not isinstance(error, dict):
        return False
    code = str(error.get("code", "")).lower()
    message = str(error.get("message", "")).lower()
    return code in {
        "rate_limit_exceeded",
        "overloaded",
        "overloaded_error",
        "server_overloaded",
    } or (
        "overloaded" in message and ("try again" in message or "temporarily" in message)
    )


def _has_generation(value):
    if not isinstance(value, dict):
        return False
    if value.get("output") or value.get("choices"):
        return True
    usage = value.get("usage")
    if isinstance(usage, dict) and any(
        type(usage.get(key)) is int and usage[key] > 0
        for key in (
            "input_tokens",
            "prompt_tokens",
            "output_tokens",
            "completion_tokens",
            "total_tokens",
        )
    ):
        return True
    return False


def _probe_error(raw, content_type, *, complete):
    """Return retry/forward/wait without replaying a generated response."""
    if "text/event-stream" not in content_type:
        if not complete:
            return "wait"
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            return "forward"
        if (
            isinstance(value, dict)
            and not _has_generation(value)
            and not _has_generation(value.get("response"))
            and _overload(value.get("error"))
        ):
            return "retry"
        return "forward"
    normalised = raw.replace(b"\r\n", b"\n")
    frames = normalised.split(b"\n\n")
    if not complete:
        frames = frames[:-1]
    rejected = False
    for frame in frames:
        data = b"\n".join(
            line[5:].lstrip()
            for line in frame.split(b"\n")
            if line.startswith(b"data:")
        )
        if not data:
            continue
        if data == b"[DONE]":
            return "retry" if rejected else "forward"
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError):
            return "forward"
        if not isinstance(value, dict):
            return "forward"
        response = value.get("response")
        if _has_generation(value) or _has_generation(response):
            return "forward"
        kind = value.get("type")
        if kind in ("response.created", "response.in_progress"):
            continue
        error = value.get("error")
        if not isinstance(error, dict) and kind == "error":
            error = value
        if not isinstance(error, dict) and isinstance(response, dict):
            error = response.get("error")
        if _overload(error) and kind in (None, "error", "response.failed"):
            rejected = True
            continue
        return "forward"
    return ("retry" if rejected else "forward") if complete else "wait"


def _retry_after(value, now):
    if not value:
        return 0.0
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


def _rate_limit_reset(value):
    """Read exhausted quota reset delays from HF's RateLimit header."""
    delays = []
    for quota in (value or "").split(","):
        remaining = re.search(r"(?:^|;)\s*r\s*=\s*(\d{1,12})\s*(?:;|$)", quota)
        reset = re.search(r"(?:^|;)\s*t\s*=\s*(\d{1,12})\s*(?:;|$)", quota)
        if remaining and reset and int(remaining.group(1)) == 0:
            delays.append(float(reset.group(1)))
    return max(delays, default=0.0)


class _ResponseOutcome:
    """Inspect bounded response frames without changing forwarded bytes."""

    def __init__(self, content_type, route="/responses"):
        self.streaming = "text/event-stream" in content_type
        self.chat = route == "/chat/completions"
        self.buffer = b""
        self.failed = False
        self.oversized = False
        self.terminal = False

    def feed(self, chunk):
        self.buffer += chunk
        if self.streaming:
            self.buffer = self.buffer.replace(b"\r\n", b"\n")
            while b"\n\n" in self.buffer:
                frame, self.buffer = self.buffer.split(b"\n\n", 1)
                if not self.oversized:
                    self.inspect(b"\n".join(
                        line[5:].lstrip() for line in frame.split(b"\n")
                        if line.startswith(b"data:")
                    ))
                self.oversized = False
        if len(self.buffer) > PROBE_BYTES:
            self.buffer = self.buffer[-1:] if self.streaming else b""
            self.oversized = True

    def inspect(self, raw):
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            return
        if not isinstance(value, dict):
            return
        candidates = [value]
        if isinstance(value.get("response"), dict):
            candidates.append(value["response"])
        for item in candidates:
            status = item.get("status")
            self.failed |= item.get("error") is not None or item.get("type") in (
                "error", "response.failed", "response.cancelled"
            ) or status in ("failed", "cancelled")
            if not self.chat:
                self.terminal |= status in ("completed", "incomplete", "failed", "cancelled")
            choices = item.get("choices")
            if isinstance(choices, list):
                for choice in choices[:64]:
                    if not isinstance(choice, dict):
                        continue
                    reason = choice.get("finish_reason")
                    if self.chat and isinstance(reason, str) and reason:
                        self.terminal = True
                    self.failed |= reason in ("error", "failed", "cancelled")

    def finish(self):
        if not self.streaming and not self.oversized:
            self.inspect(self.buffer)
        return self.failed or (self.streaming and not self.terminal)


class RelayState:
    def __init__(
        self, config, *, directory, opener=None, now=None, sleep=None, jitter=None
    ):
        self.model = config["model"]
        self.marker = config["marker"]
        self.key = config["api_key"]
        self.base_url = config["base_url"].rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("A fixed HTTPS provider endpoint is required")
        self.maximum_requests = config["maximum_requests"]
        from simready_bench.engine.native_output import DEFAULT_AUTHORING_TURNS

        self.maximum_condition_requests = config.get(
            "maximum_condition_requests", min(DEFAULT_AUTHORING_TURNS, self.maximum_requests)
        )
        self.expires_at = config["expires_at"]
        self.maximum_attempts = config.get("maximum_attempts", 4)
        self.maximum_retry_seconds = config.get("maximum_retry_seconds", 120)
        self.native_output_only = config.get("native_output_only", True)
        self.maximum_request_bytes = config.get("maximum_request_bytes", 32 * 1024**2)
        self.sandbox_id = config.get("sandbox_id")
        if (
            type(self.maximum_requests) is not int
            or self.maximum_requests < 1
            or type(self.expires_at) not in (int, float)
            or not math.isfinite(self.expires_at)
            or type(self.native_output_only) is not bool
            or type(self.maximum_request_bytes) is not int
            or not 0 < self.maximum_request_bytes <= 64 * 1024**2
            or type(self.maximum_condition_requests) is not int
            or self.maximum_condition_requests < 1
            or type(self.maximum_attempts) is not int
            or not 1 <= self.maximum_attempts <= 8
            or type(self.maximum_retry_seconds) not in (int, float)
            or not math.isfinite(self.maximum_retry_seconds)
            or not 0 < self.maximum_retry_seconds <= 300
            or self.sandbox_id is not None
            and (
                not isinstance(self.sandbox_id, str)
                or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", self.sandbox_id)
            )
        ):
            raise ValueError("Bounded provider retry settings required")
        self.project_id = config.get("project_id")
        self.shared_route_throttle = None
        throttle_config = config.get("shared_route_throttle")
        if throttle_config is not None:
            if (
                not isinstance(throttle_config, dict)
                or set(throttle_config) - {"url", "token", "route_key", "preview_token"}
                or not {"url", "token", "route_key"} <= set(throttle_config)
                or any(not isinstance(value, str) or not value for value in throttle_config.values())
            ):
                raise ValueError("A fixed shared route throttle configuration is required")
            from simready_bench.route_throttle import RouteThrottleClient

            self.shared_route_throttle = RouteThrottleClient(**throttle_config)
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.now = now or time.time
        self.sleep = sleep or time.sleep
        self.jitter = jitter or (
            lambda delay: random.uniform(0.75 * delay, 1.25 * delay)
        )
        self.lock = threading.Lock()
        self.requests = 0
        self.counters = 0
        self.events = []
        self.scope_requests = {}
        self.scope_counters = {}
        self.scope_limits = {}
        self.calls = 0
        self.scope_calls = {}
        self.maximum_transport_requests = self.maximum_requests * self.maximum_attempts
        ledger = self.directory / "ledger.json"
        if ledger.exists():
            if (
                ledger.is_symlink()
                or not ledger.is_file()
                or ledger.stat().st_size > 32 * 1024**2
            ):
                raise ValueError("A bounded existing relay ledger is required")
            previous = json.loads(ledger.read_text(encoding="utf-8"))
            if (
                any(
                    previous.get(key) != value
                    for key, value in (
                        ("model", self.model),
                        ("maximum_requests", self.maximum_requests),
                        ("expires_at", self.expires_at),
                    )
                )
                or previous.get(
                    "maximum_condition_requests", self.maximum_condition_requests
                )
                != self.maximum_condition_requests
                or previous.get("maximum_transport_requests", self.maximum_transport_requests)
                != self.maximum_transport_requests
            ):
                raise ValueError("The existing relay allowance cannot change")
            self.requests = previous["generation_requests"]
            self.counters = previous["input_counter_requests"]
            self.events = previous["events"]
            self.scope_requests = previous.get("condition_generation_requests", {})
            self.scope_counters = previous.get("condition_input_counter_requests", {})
            self.scope_limits = previous.get("condition_limits", {})
            if (
                type(self.requests) is not int
                or not 0 <= self.requests <= self.maximum_transport_requests
                or type(self.counters) is not int
                or not 0 <= self.counters <= 4 * self.maximum_requests
            ):
                raise ValueError("The existing relay counters are invalid")
            if any(
                not isinstance(value, dict)
                for value in (
                    self.scope_requests,
                    self.scope_counters,
                    self.scope_limits,
                )
            ):
                raise ValueError("The existing condition counters are invalid")
            for scope, limit in self.scope_limits.items():
                if (
                    not isinstance(scope, str)
                    or not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", scope)
                    or type(limit) is not int
                    or not 1 <= limit <= self.maximum_condition_requests
                ):
                    raise ValueError("The existing condition allowance is invalid")
            for counts, multiplier in (
                (self.scope_requests, self.maximum_attempts),
                (self.scope_counters, 4),
            ):
                if any(
                    scope not in self.scope_limits
                    or type(count) is not int
                    or not 0 <= count <= multiplier * self.scope_limits[scope]
                    for scope, count in counts.items()
                ):
                    raise ValueError("The existing condition counters are invalid")
            if (
                sum(self.scope_requests.values()) > self.requests
                or sum(self.scope_counters.values()) > self.counters
            ):
                raise ValueError("The existing relay counters are inconsistent")
            # Older ledgers counted attempts as calls. Preserve their authority
            # conservatively rather than refunding historical, ambiguous outcomes.
            self.calls = previous.get("model_calls", self.requests)
            self.scope_calls = previous.get("condition_model_calls", dict(self.scope_requests))
            if (
                type(self.calls) is not int
                or not 0 <= self.calls <= self.maximum_requests
                or not isinstance(self.scope_calls, dict)
                or any(
                    scope not in self.scope_limits
                    or type(count) is not int
                    or not 0 <= count <= self.scope_limits[scope]
                    for scope, count in self.scope_calls.items()
                )
                or sum(self.scope_calls.values()) > self.calls
            ):
                raise ValueError("The existing model call counters are invalid")
        self.save()

    def save(self):
        value = {
            "model": self.model,
            "generation_requests": self.requests,
            "input_counter_requests": self.counters,
            "maximum_requests": self.maximum_requests,
            "maximum_condition_requests": self.maximum_condition_requests,
            "expires_at": self.expires_at,
            "events": self.events,
            "condition_generation_requests": self.scope_requests,
            "condition_input_counter_requests": self.scope_counters,
            "condition_limits": self.scope_limits,
            "model_calls": self.calls,
            "condition_model_calls": self.scope_calls,
            "maximum_transport_requests": self.maximum_transport_requests,
            "accounting_version": 2,
        }
        temporary = self.directory / "ledger.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.directory / "ledger.json")

    def admit(self, route, *, scope=None, limit=None, deadline=None):
        with self.lock:
            if self.now() >= min(self.expires_at, deadline or self.expires_at):
                raise RelayAllowanceError("Model authority has expired")
            counter = route == "/responses/input_tokens"
            if (counter and self.counters >= 4 * self.maximum_requests) or (
                not counter and self.calls >= self.maximum_requests
            ):
                raise RelayAllowanceError("Model request allowance is exhausted")
            if not counter and self.requests >= self.maximum_transport_requests:
                raise RelayAllowanceError("Model transport attempt allowance is exhausted")
            if scope is not None:
                if self.requests != sum(self.scope_requests.values()):
                    raise RelayAllowanceError(
                        "Existing unscoped requests require their original relay protocol"
                    )
                if not isinstance(scope, str) or not re.fullmatch(
                    r"[A-Za-z0-9._-]{1,160}", scope
                ):
                    raise RelayAllowanceError("A literal condition allowance is required")
                if (
                    type(limit) is not int
                    or not 1 <= limit <= self.maximum_condition_requests
                ):
                    raise RelayAllowanceError(
                        "A bounded condition request allowance is required"
                    )
                if scope in self.scope_limits and self.scope_limits[scope] != limit:
                    raise RelayAllowanceError("A condition request allowance cannot change")
                self.scope_limits[scope] = limit
                counts = self.scope_counters if counter else self.scope_requests
                if (counter and counts.get(scope, 0) >= 4 * limit) or (
                    not counter and self.scope_calls.get(scope, 0) >= limit
                ):
                    raise RelayAllowanceError("Condition request allowance is exhausted")
                if not counter and counts.get(scope, 0) >= limit * self.maximum_attempts:
                    raise RelayAllowanceError("Condition transport attempt allowance is exhausted")
                counts[scope] = counts.get(scope, 0) + 1
            if counter:
                self.counters += 1
            else:
                self.requests += 1
                self.calls += 1
                if scope is not None:
                    self.scope_calls[scope] = self.scope_calls.get(scope, 0) + 1
            ordinal = self.counters if counter else self.requests
            request_limit = 4 * self.maximum_requests if counter else self.maximum_requests
            event = {
                "path": route,
                "started_at": self.now(),
                "status": "admitted",
                "request_kind": "input_token_count" if counter else "generation",
                "request_ordinal": ordinal,
                "request_limit": request_limit,
                "model_call_reserved": not counter,
            }
            if scope is not None:
                event["condition_scope"] = scope
                event["condition_request_ordinal"] = counts[scope]
                event["condition_request_limit"] = 4 * limit if counter else limit
            self.events.append(event)
            self.save()
            return event

    def finish(self, event, status, **fields):
        with self.lock:
            if status != "completed" and event.get("model_call_reserved"):
                self.calls -= 1
                scope = event.get("condition_scope")
                if scope is not None:
                    self.scope_calls[scope] -= 1
                event["model_call_reserved"] = False
            event.update(status=status, finished_at=self.now(), **fields)
            self.save()

    def capacity(self, route, scope, limit):
        with self.lock:
            if route == "/responses/input_tokens":
                return (
                    self.counters < 4 * self.maximum_requests
                    and self.scope_counters.get(scope, 0) < 4 * limit
                )
            return (
                self.calls < self.maximum_requests
                and self.scope_calls.get(scope, 0) < limit
                and self.requests < self.maximum_transport_requests
                and self.scope_requests.get(scope, 0) < limit * self.maximum_attempts
            )


class Relay(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _emit_event(self, state, name, event, *, attempt=None, **extra):
        now = state.now()
        scope = event.get("condition_scope")
        started_at = event.get("started_at", now)
        value = {
            "event": name,
            "timestamp": now,
            "sandbox_id": state.sandbox_id,
            "model": state.model,
            "route": event.get("path"),
            "request_kind": event.get("request_kind"),
            "request_ordinal": event.get("request_ordinal"),
            "request_limit": event.get("request_limit"),
            "condition_request_ordinal": event.get("condition_request_ordinal"),
            "condition_request_limit": event.get("condition_request_limit"),
            "condition_scope_hash": (
                hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
                if scope is not None
                else None
            ),
            "status": event.get("status"),
            "http_status": event.get("http_status"),
            "elapsed_seconds": max(0, now - started_at),
        }
        if attempt is not None:
            value["attempt"] = attempt
        value.update(extra)
        try:
            print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
        except Exception:
            pass

    def _prefix(self, response, state, deadline, lease=None):
        content_type = response.headers.get("Content-Type", "").lower()
        if response.status not in RETRYABLE_STATUS and (
            response.status != 200
            or not any(kind in content_type for kind in ("json", "text/event-stream"))
        ):
            return b"", False
        prefix = b""
        sock = getattr(
            getattr(getattr(response, "fp", None), "raw", None), "_sock", None
        )
        while len(prefix) <= PROBE_BYTES:
            if lease is not None:
                lease.ensure_active()
            remaining = deadline - state.now()
            if remaining <= 0:
                raise TimeoutError("Model request deadline reached")
            if sock is not None and sock.fileno() >= 0:
                sock.settimeout(min(300, remaining, lease.lease_seconds / 2 if lease else 300))
            chunk = getattr(response, "read1", response.read)(
                min(4096, PROBE_BYTES + 1 - len(prefix))
            )
            if chunk:
                prefix += chunk
            decision = _probe_error(prefix, content_type, complete=not chunk)
            if response.status in RETRYABLE_STATUS:
                if not chunk:
                    if "text/event-stream" in content_type:
                        return prefix, decision == "retry"
                    try:
                        value = json.loads(prefix)
                        generated = _has_generation(value) or (
                            isinstance(value, dict)
                            and _has_generation(value.get("response"))
                        )
                    except (ValueError, UnicodeError):
                        generated = False
                    return prefix, not generated
            elif decision != "wait":
                return prefix, decision == "retry"
            if not chunk:
                break
        return prefix, False

    def do_POST(self):
        state = self.server.state
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + state.marker
        ):
            self.send_error(403)
            return
        if self.path not in (
            "/v1/responses",
            "/v1/responses/input_tokens",
            "/v1/responses/compact",
            "/v1/chat/completions",
        ):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= state.maximum_request_bytes:
                raise ValueError("Bounded model request required")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete model request")
            body = json.loads(raw)
            if body.get("model") != state.model:
                raise ValueError("Model differs from the authorised route")
            if state.native_output_only and any(
                field in body
                for field in (
                    "max_tokens",
                    "max_output_tokens",
                    "max_completion_tokens",
                )
            ):
                raise ValueError("Provider-native output limits required")
            route = self.path.removeprefix("/v1")
            scope = self.headers.get("X-SimReady-Condition")
            limit = (
                int(self.headers["X-SimReady-Request-Limit"])
                if scope is not None
                else None
            )
            deadline = min(
                state.expires_at,
                float(self.headers.get("X-SimReady-Deadline", state.expires_at)),
            )
            if not math.isfinite(deadline) or deadline <= state.now():
                raise ValueError("A live bounded request deadline is required")
        except (ValueError, TypeError):
            self.send_error(400)
            return
        response = None
        event = None
        sent_headers = False
        lease = None
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + state.key,
            }
            if state.project_id:
                headers["OpenAI-Project"] = state.project_id
            request = urllib.request.Request(
                state.base_url + route, data=raw, headers=headers
            )
            started = state.now()
            attempts = 0
            while True:
                event = None
                if state.shared_route_throttle is not None:
                    lease = state.shared_route_throttle.acquire(
                        timeout=min(300, max(0.001, deadline - state.now()))
                    )
                event = state.admit(route, scope=scope, limit=limit, deadline=deadline)
                attempts += 1
                self._emit_event(
                    state, "model_request_started", event, attempt=attempts
                )
                if lease is not None:
                    lease.ensure_active()
                try:
                    response = state.opener.open(
                        request, timeout=min(
                            300, max(0.001, deadline - state.now()),
                            lease.lease_seconds / 2 if lease else 300,
                        )
                    )
                except urllib.error.HTTPError as error:
                    response = error
                if lease is not None:
                    lease.set_on_lost(response.close)
                    lease.ensure_active()
                event["http_status"] = response.status
                for header, field in (
                    ("Retry-After", "retry_after"),
                    ("RateLimit", "rate_limit"),
                    ("RateLimit-Policy", "rate_limit_policy"),
                ):
                    if response.headers.get(header):
                        event[field] = response.headers[header][:1024]
                prefix, rejected = self._prefix(response, state, deadline, lease)
                delay = max(
                    state.jitter(min(30, 2 ** (attempts - 1))),
                    _retry_after(response.headers.get("Retry-After"), state.now()),
                    _rate_limit_reset(response.headers.get("RateLimit")),
                )
                if rejected:
                    state.finish(event, "retryable_rejection", retry_delay_seconds=delay)
                if not (
                    rejected
                    and scope is not None
                    and state.capacity(route, scope, limit)
                    and attempts < state.maximum_attempts
                    and state.now() + delay
                    < min(deadline, started + state.maximum_retry_seconds)
                ):
                    break
                self._emit_event(
                    state,
                    "model_request_retrying",
                    event,
                    attempt=attempts,
                    retry_delay_seconds=delay,
                )
                if lease is not None:
                    lease.finish(status_code=response.status, headers=dict(response.headers))
                    lease = None
                response.close()
                response = None
                state.sleep(delay)
            self.send_response(response.status)
            for name in (
                "Content-Type",
                "Content-Length",
                "Retry-After",
                "x-request-id",
                "openai-project",
                "openai-organization",
            ):
                if response.headers.get(name):
                    self.send_header(name, response.headers[name])
            self.send_header("X-SimReady-Upstream-Attempts", str(attempts))
            if scope is not None:
                self.send_header(
                    "X-SimReady-Scope-Requests", str(state.scope_requests.get(scope, 0))
                )
            self.end_headers()
            sent_headers = True
            outcome = _ResponseOutcome(response.headers.get("Content-Type", ""), route)
            outcome.feed(prefix)
            if prefix:
                self.wfile.write(prefix)
                self.wfile.flush()
            sock = getattr(
                getattr(getattr(response, "fp", None), "raw", None), "_sock", None
            )
            streaming_started = time.monotonic()
            while True:
                if lease is not None:
                    lease.ensure_active()
                remaining = min(
                    1800 - (time.monotonic() - streaming_started),
                    deadline - state.now(),
                )
                if remaining <= 0:
                    raise TimeoutError("Model request deadline reached")
                if sock is not None and sock.fileno() >= 0:
                    sock.settimeout(min(300, remaining, lease.lease_seconds / 2 if lease else 300))
                chunk = getattr(response, "read1", response.read)(4096)
                if not chunk:
                    break
                outcome.feed(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            if lease is not None:
                lease.ensure_active()
            failed = rejected or not 200 <= response.status < 300 or outcome.finish()
            state.finish(event, "inference_failed" if failed else "completed")
            self._emit_event(
                state, "model_request_finished", event, attempt=attempts
            )
        except (BrokenPipeError, ConnectionResetError):
            if event is not None:
                state.finish(event, "client_disconnected")
                self._emit_event(
                    state, "model_request_failed", event, attempt=attempts
                )
        except RelayAllowanceError as error:
            if event is not None:
                state.finish(
                    event, "allowance_exhausted", error_type=type(error).__name__
                )
                self._emit_event(
                    state,
                    "model_request_failed",
                    event,
                    attempt=attempts,
                    error_type=type(error).__name__,
                )
            if not sent_headers:
                body = json.dumps({"error": {"code": "relay_allowance_exhausted", "type": "invalid_request_error", "message": str(error)}}).encode("utf-8")
                self.send_response(409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as error:
            if event is not None:
                state.finish(event, "transport_error", error_type=type(error).__name__)
                self._emit_event(
                    state,
                    "model_request_failed",
                    event,
                    attempt=attempts,
                    error_type=type(error).__name__,
                )
            if not sent_headers:
                self.send_error(502)
        finally:
            try:
                if lease is not None:
                    lease.finish(
                        status_code=response.status if response is not None else None,
                        headers=dict(response.headers) if response is not None else None,
                    )
            finally:
                if response is not None:
                    response.close()


def main(path):
    envelope = Path(path)
    config = json.loads(envelope.read_text(encoding="utf-8"))
    envelope.unlink()
    state = RelayState(config, directory=envelope.parent)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.daemon_threads = True
    server.state = state
    ready = envelope.parent / "ready.json"
    ready.write_text(json.dumps({"port": server.server_port}) + "\n", encoding="utf-8")
    ready.chmod(0o600)
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1])
