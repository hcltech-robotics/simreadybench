"""Authenticated, durable admission control shared by inference route workers.

This module uses only the standard library so cloud relays can carry the same
client without installing the benchmark package. Tokens are supplied through a
private configuration file, never through command line arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import random
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import HTTPError, URLError


def route_identity(origin: str, provider: str, model: str, credential_identity: str) -> str:
    """Hash the route and account identity without retaining credential material."""
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A valid inference origin is required")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Inference origin cannot contain credentials or query parameters")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    canonical = [parsed.scheme.lower(), parsed.hostname.lower(), port, provider, model,
                 hashlib.sha256(credential_identity.encode()).hexdigest()]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def _retry_delay(headers: Mapping[str, str], now: float) -> float | None:
    """Read explicit server cooldown hints, including HF structured RateLimit."""
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    delays: list[float] = []
    retry_after = lowered.get("retry-after", "")
    if retry_after:
        try:
            delays.append(float(retry_after))
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(retry_after).timestamp() - now)
            except (TypeError, ValueError, OverflowError):
                pass
    for key in ("ratelimit-reset", "x-ratelimit-reset"):
        try:
            value = float(lowered[key])
            # X-RateLimit-Reset commonly uses epoch seconds. RateLimit-Reset
            # commonly uses a relative delay. Accept both unambiguously.
            delays.append(value - now if value > 1_000_000_000 else value)
        except (KeyError, ValueError):
            pass
    for match in re.finditer(r"(?:^|[;,])\s*t\s*=\s*(\d+(?:\.\d+)?)", lowered.get("ratelimit", "")):
        delays.append(float(match.group(1)))
    usable = [value for value in delays if math.isfinite(value) and value >= 0]
    return max(usable) if usable else None


class RouteThrottleStore:
    """SQLite transactions serialise admission across service threads and processes."""

    def __init__(self, database: str | Path, *, max_concurrency: int = 4,
                 lease_seconds: float = 120, clock=time.time, jitter=random.uniform):
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if not math.isfinite(lease_seconds) or lease_seconds < 1:
            raise ValueError("lease_seconds must be at least one second")
        self.database = str(database)
        self.max_concurrency = max_concurrency
        self.lease_seconds = float(lease_seconds)
        self.clock = clock
        self.jitter = jitter
        Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS routes (
                    route TEXT PRIMARY KEY, cooldown REAL NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS leases (
                    lease TEXT PRIMARY KEY, route TEXT NOT NULL, expires REAL NOT NULL,
                    request TEXT NOT NULL UNIQUE);
                CREATE INDEX IF NOT EXISTS leases_route ON leases(route, expires);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=15, isolation_level=None)
        db.execute("PRAGMA busy_timeout=15000")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _route(route: str) -> None:
        if not isinstance(route, str) or not re.fullmatch(r"[0-9a-f]{64}", route):
            raise ValueError("route must be a SHA-256 identity")

    @staticmethod
    def _lease(lease: str) -> None:
        if not isinstance(lease, str) or not re.fullmatch(r"[0-9a-f]{32}", lease):
            raise ValueError("lease must be a UUID hex string")

    def acquire(self, route: str, request_id: str | None = None) -> dict:
        self._route(route)
        request_id = request_id or uuid.uuid4().hex
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise ValueError("request_id must be a UUID hex string")
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM leases WHERE expires <= ?", (now,))
            existing = db.execute("SELECT lease, route, expires FROM leases WHERE request=?", (request_id,)).fetchone()
            if existing:
                if existing[1] != route:
                    raise ValueError("request_id already belongs to another route")
                expires = now + self.lease_seconds
                db.execute("UPDATE leases SET expires=? WHERE lease=?", (expires, existing[0]))
                db.commit()
                return {"admitted": True, "lease": existing[0], "expires_at": expires,
                        "lease_seconds": self.lease_seconds}
            db.execute("INSERT OR IGNORE INTO routes(route) VALUES (?)", (route,))
            cooldown = db.execute("SELECT cooldown FROM routes WHERE route=?", (route,)).fetchone()[0]
            count = db.execute("SELECT count(*) FROM leases WHERE route=?", (route,)).fetchone()[0]
            if cooldown > now or count >= self.max_concurrency:
                db.commit()
                return {"admitted": False, "retry_after": max(0.25, cooldown - now, 1.0 if count >= self.max_concurrency else 0)}
            lease = uuid.uuid4().hex
            expires = now + self.lease_seconds
            db.execute("INSERT INTO leases VALUES (?,?,?,?)", (lease, route, expires, request_id))
            db.commit()
            return {"admitted": True, "lease": lease, "expires_at": expires,
                    "lease_seconds": self.lease_seconds}

    def heartbeat(self, route: str, lease: str) -> dict:
        self._route(route)
        self._lease(lease)
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("UPDATE leases SET expires=? WHERE lease=? AND route=? AND expires>?",
                                (now + self.lease_seconds, lease, route, now))
            db.commit()
            return {"renewed": cursor.rowcount == 1, "expires_at": now + self.lease_seconds}

    def finish(self, route: str, lease: str, *, status_code: int | None = None,
               headers: Mapping[str, str] | None = None) -> dict:
        self._route(route)
        self._lease(lease)
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # An expired lease may still report a real upstream 429. Keep that
            # route cooldown, but never allow repeated finish to extend it.
            row = db.execute("SELECT route FROM leases WHERE lease=?", (lease,)).fetchone()
            if row is None or row[0] != route:
                db.commit()
                return {"released": False}
            if status_code == 429:
                cooldown, failures = db.execute("SELECT cooldown, failures FROM routes WHERE route=?", (route,)).fetchone()
                delay = _retry_delay(headers or {}, now)
                if delay is None:
                    delay = min(60.0, 2.0 ** min(failures + 1, 6)) * self.jitter(1.0, 1.5)
                db.execute("UPDATE routes SET cooldown=?, failures=? WHERE route=?",
                           (max(cooldown, now + delay), failures + 1, route))
            elif status_code is not None and 200 <= status_code < 300:
                db.execute("UPDATE routes SET failures=0 WHERE route=? AND cooldown<=?", (route, now))
            db.execute("DELETE FROM leases WHERE lease=? AND route=?", (lease, route))
            db.commit()
            return {"released": True}


class RouteThrottleServer:
    """Small authenticated HTTP service. All scheduling state survives restart."""

    def __init__(self, database: str | Path, token: str, *, host: str = "127.0.0.1",
                 port: int = 0, max_concurrency: int = 4, lease_seconds: float = 120,
                 route_keys: list[str] | None = None):
        if not isinstance(token, str) or len(token) < 16 or not token.isascii() or "\n" in token or "\r" in token:
            raise ValueError("A private ASCII bearer token of at least 16 characters is required")
        if not route_keys:
            raise ValueError("The service requires a fixed nonempty route_keys registry")
        for route in route_keys:
            RouteThrottleStore._route(route)
        allowed_routes = frozenset(route_keys)
        self.store = RouteThrottleStore(database, max_concurrency=max_concurrency, lease_seconds=lease_seconds)
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _reply(self, status, payload):
                body = json.dumps(payload, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.connection.settimeout(10)
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                    self._reply(401, {"error": "unauthorised"})
                    return
                try:
                    if self.headers.get("Transfer-Encoding"):
                        raise ValueError("Transfer encoding is unsupported")
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 16384:
                        raise ValueError("Body must be between 1 and 16384 bytes")
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError("Body must be an object")
                    route = payload["route"]
                    if route not in allowed_routes:
                        self._reply(403, {"error": "route_not_authorised"})
                        return
                    if self.path == "/acquire":
                        result = store.acquire(route, payload["request_id"])
                    elif self.path == "/heartbeat":
                        result = store.heartbeat(route, payload["lease"])
                    elif self.path == "/finish":
                        status = payload.get("status_code")
                        headers = payload.get("headers", {})
                        if status is not None and (isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599):
                            raise ValueError("Invalid status_code")
                        if not isinstance(headers, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()):
                            raise ValueError("Invalid headers")
                        result = store.finish(route, payload["lease"], status_code=status, headers=headers)
                    else:
                        self._reply(404, {"error": "unknown_endpoint"})
                        return
                    self._reply(200, result)
                except (KeyError, ValueError, TypeError, OverflowError):
                    self._reply(400, {"error": "invalid_request"})

            def do_GET(self):
                if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
                    self._reply(401, {"error": "unauthorised"})
                elif self.path == "/health":
                    self._reply(200, {"ok": True})
                else:
                    self._reply(404, {"error": "unknown_endpoint"})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{port}"

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self._thread.start()
        return self

    def close(self):
        if self._thread is not None:
            self.httpd.shutdown()
            self._thread.join(timeout=15)
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.close()


class RouteThrottleClient:
    def __init__(self, url: str, token: str, route_key: str, *, timeout: float = 10,
                 preview_token: str | None = None):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Invalid throttle service URL")
        RouteThrottleStore._route(route_key)
        self.url = url.rstrip("/")
        self.token = token
        self.route_key = route_key
        self.timeout = timeout
        self.preview_token = preview_token
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, _request, _fp, _code, _message, _headers, _newurl):
                return None
        self._opener = build_opener(NoRedirect())

    def _post(self, endpoint: str, payload: dict, *, timeout: float | None = None) -> dict:
        body = json.dumps({"route": self.route_key, **payload}, separators=(",", ":")).encode()
        headers = {"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}
        if self.preview_token:
            headers["X-Daytona-Preview-Token"] = self.preview_token
        request = Request(self.url + endpoint, data=body, headers=headers)
        with self._opener.open(request, timeout=timeout if timeout is not None else self.timeout) as response:
            return json.loads(response.read(16385))

    def acquire(self, *, timeout: float = 300) -> ThrottleLease:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("An acquire timeout greater than zero is required")
        deadline = time.monotonic() + timeout
        request_id = uuid.uuid4().hex
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Shared inference route admission deadline exceeded")
            try:
                result = self._post("/acquire", {"request_id": request_id}, timeout=min(self.timeout, remaining))
            except HTTPError as exc:
                if exc.code < 500:
                    raise
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                continue
            except (URLError, TimeoutError):
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                continue
            if result.get("admitted"):
                return ThrottleLease(self, result["lease"], result["lease_seconds"])
            delay = min(max(0.25, float(result["retry_after"])), remaining)
            time.sleep(delay)


class ThrottleLease:
    def __init__(self, client: RouteThrottleClient, lease: str, lease_seconds: float):
        self.client = client
        self.lease = lease
        self.lease_seconds = float(lease_seconds)
        self.heartbeat_error: Exception | None = None
        self._finished = False
        self._on_lost = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._renew, daemon=True)
        self._thread.start()

    def _renew(self):
        interval = max(0.1, self.lease_seconds / 3)
        while not self._stop.wait(interval):
            try:
                result = self.client._post("/heartbeat", {"lease": self.lease}, timeout=min(self.client.timeout, interval))
                if not result.get("renewed"):
                    raise RuntimeError("Shared inference route lease expired")
                self.heartbeat_error = None
            except Exception as exc:
                self.heartbeat_error = exc
                callback = self._on_lost
                if callback is not None:
                    try:
                        callback()
                    except Exception:
                        pass
                return

    def set_on_lost(self, callback):
        """Close the upstream stream immediately if central lease ownership is lost."""
        self._on_lost = callback
        if self.heartbeat_error is not None:
            callback()

    def ensure_active(self):
        if self.heartbeat_error is not None:
            raise RuntimeError("Shared inference route lease heartbeat failed") from self.heartbeat_error
        if self._finished:
            raise RuntimeError("Shared inference route lease has been released")

    def finish(self, *, status_code: int | None = None, headers: Mapping[str, str] | None = None):
        try:
            with self._lock:
                if self._finished:
                    return
                # No provider credentials, cookies or arbitrary response bodies
                # pass through the central service.
                safe = {key: str(value) for key, value in (headers or {}).items()
                        if key.lower() in {"retry-after", "ratelimit-reset", "x-ratelimit-reset", "ratelimit"}}
                self._finished = True
                self.client._post("/finish", {"lease": self.lease, "status_code": status_code, "headers": safe})
        finally:
            # If release cannot reach the service, stop renewing. The durable
            # lease then expires instead of leaking capacity indefinitely.
            self._stop.set()
            self._thread.join(timeout=self.client.timeout + 1)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expires_at = config.pop("expires_at", None)
    if expires_at is not None and (isinstance(expires_at, bool) or not isinstance(expires_at, (float, int)) or not math.isfinite(expires_at)):
        raise ValueError("expires_at must be a finite Unix timestamp")
    server = RouteThrottleServer(**config)
    with server:
        try:
            while expires_at is None or time.time() < expires_at:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
