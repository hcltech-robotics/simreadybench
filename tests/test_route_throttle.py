"""Local HTTP proof of shared route admission, cooldown and recovery."""

from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from simready_bench.route_throttle import (
    RouteThrottleClient,
    RouteThrottleServer,
    RouteThrottleStore,
    _retry_delay,
    route_identity,
)


TOKEN = "local-test-token-at-least-16"
ROUTE_A = route_identity("https://router.huggingface.co", "baseten", "model-a", "credential-one")
ROUTE_B = route_identity("https://router.huggingface.co", "baseten", "model-b", "credential-one")


@pytest.fixture
def server(tmp_path):
    with RouteThrottleServer(tmp_path / "throttle.sqlite", TOKEN, max_concurrency=2,
                             lease_seconds=2, route_keys=[ROUTE_A, ROUTE_B]) as service:
        yield service


def client(server, route=ROUTE_A, **kwargs):
    return RouteThrottleClient(server.url, TOKEN, route, **kwargs)


def test_identity_binds_route_and_account_without_credentials():
    original = route_identity("https://router.huggingface.co", "baseten", "model-a", "credential-one")
    assert original == route_identity("https://ROUTER.huggingface.co:443/v1", "baseten", "model-a", "credential-one")
    assert original != route_identity("https://other.example", "baseten", "model-a", "credential-one")
    assert original != route_identity("https://router.huggingface.co", "other", "model-a", "credential-one")
    assert original != route_identity("https://router.huggingface.co", "baseten", "model-a", "credential-two")
    assert "credential" not in original
    with pytest.raises(ValueError):
        route_identity("https://secret@example.com", "p", "m", "c")


def test_live_limit_changes_preserve_leases_and_cooldown(tmp_path):
    now = [100.0]
    database = tmp_path / 'limits.sqlite'
    one = RouteThrottleStore(database, clock=lambda: now[0])
    two = RouteThrottleStore(database, clock=lambda: now[0])
    leases = [one.acquire(ROUTE_A, str(index).zfill(32)) for index in range(4)]
    assert all(lease['admitted'] for lease in leases)
    assert not two.acquire(ROUTE_A, 'a' * 32)['admitted']
    one.set_limit(ROUTE_A, 12)
    extra = [two.acquire(ROUTE_A, str(index).zfill(32)) for index in range(4, 12)]
    assert all(lease['admitted'] for lease in extra)
    assert not one.acquire(ROUTE_A, 'a' * 32)['admitted']
    two.set_limit(ROUTE_A, 2)
    assert not one.acquire(ROUTE_A, 'a' * 32)['admitted']
    assert one.acquire(ROUTE_A, '0' * 32)['lease'] == leases[0]['lease']
    with one._connect() as db:
        db.execute('UPDATE routes SET cooldown=200 WHERE route=?', (ROUTE_A,))
        db.execute('DELETE FROM leases')
    one.set_limit(ROUTE_A, 12)
    assert two.acquire(ROUTE_A, 'a' * 32)['retry_after'] == 100


def test_concurrent_store_schema_migration(tmp_path):
    database = tmp_path / 'migration.sqlite'
    with ThreadPoolExecutor(max_workers=8) as pool:
        stores = list(pool.map(lambda _: RouteThrottleStore(database), range(8)))
    stores[0].set_limit(ROUTE_A, 12)
    assert stores[-1].acquire(ROUTE_A, 'a' * 32)['admitted']


def test_real_http_clients_share_concurrency(server):
    active = 0
    high_water = 0
    lock = threading.Lock()

    def execute(_index):
        nonlocal active, high_water
        with client(server).acquire(timeout=10):
            with lock:
                active += 1
                high_water = max(high_water, active)
            time.sleep(0.08)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(execute, range(8)))
    assert high_water == 2
    assert active == 0


def test_real_http_independent_processes_share_concurrency(server):
    script = """
import json,sys,time
from simready_bench.route_throttle import RouteThrottleClient
with RouteThrottleClient(sys.argv[1], sys.argv[2], sys.argv[3]).acquire(timeout=10):
    started=time.monotonic()
    time.sleep(0.2)
    ended=time.monotonic()
print(json.dumps([started,ended]))
"""
    processes = [subprocess.Popen([sys.executable, "-c", script, server.url, TOKEN, ROUTE_A],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  env=os.environ.copy()) for _ in range(6)]
    events = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        started, ended = json.loads(stdout)
        events.extend([(started, 1), (ended, -1)])
    active = high_water = 0
    for _when, change in sorted(events):
        active += change
        high_water = max(high_water, active)
    assert high_water == 2


def test_authenticated_client_never_forwards_secrets_through_redirect(server):
    hits = []

    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            hits.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        do_GET = do_POST

    target = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()

    class Redirect(Receiver):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/steal")
            self.end_headers()

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    redirect_thread = threading.Thread(target=redirect.serve_forever, daemon=True)
    redirect_thread.start()
    try:
        instance = RouteThrottleClient(f"http://127.0.0.1:{redirect.server_port}", TOKEN, ROUTE_A,
                                       preview_token="private-preview")
        with pytest.raises(HTTPError) as error:
            instance.acquire(timeout=0.2)
        assert error.value.code == 302
        assert hits == []
    finally:
        redirect.shutdown()
        target.shutdown()
        redirect.server_close()
        target.server_close()
        redirect_thread.join()
        target_thread.join()


def test_route_pools_are_independent(server):
    with client(server).acquire() as first, client(server).acquire() as second:
        with client(server, ROUTE_B).acquire(timeout=0.5):
            pass
        with pytest.raises(TimeoutError, match="admission deadline"):
            client(server).acquire(timeout=0.1)
        first.ensure_active()
        second.ensure_active()


def test_real_http_429_cooldown_blocks_all_clients_only_its_route(server):
    lease = client(server).acquire()
    lease.finish(status_code=429, headers={"Retry-After": "1", "Authorization": "never-forward"})
    with client(server, ROUTE_B).acquire(timeout=0.2):
        pass
    started = time.monotonic()
    with client(server).acquire(timeout=3):
        assert time.monotonic() - started >= 0.8


@pytest.mark.parametrize("headers, expected", [
    ({"Retry-After": "12"}, 12),
    ({"retry-after": formatdate(1_700_000_020, usegmt=True)}, 20),
    ({"RateLimit-Reset": "30"}, 30),
    ({"X-RateLimit-Reset": "1700000040"}, 40),
    ({"RateLimit": '"inference";r=0;t=45'}, 45),
    ({"RateLimit": '"api";r=0;t=5, "inference";r=0;t=90'}, 90),
    ({"Retry-After": "garbage", "RateLimit": "bad"}, None),
    ({"Retry-After": "nan", "RateLimit-Reset": "inf"}, None),
])
def test_server_cooldown_hints(headers, expected):
    assert _retry_delay(headers, 1_700_000_000) == expected


def test_durable_cooldown_survives_restart_and_success_does_not_clear_it(tmp_path):
    now = [100.0]
    clock = lambda: now[0]
    path = tmp_path / "durable.sqlite"
    store = RouteThrottleStore(path, clock=clock)
    bad = store.acquire(ROUTE_A)
    good = store.acquire(ROUTE_A)
    store.finish(ROUTE_A, bad["lease"], status_code=429, headers={"Retry-After": "30"})
    store.finish(ROUTE_A, good["lease"], status_code=200)
    restarted = RouteThrottleStore(path, clock=clock)
    assert restarted.acquire(ROUTE_A) == {"admitted": False, "retry_after": 30.0}
    now[0] = 131
    assert restarted.acquire(ROUTE_A)["admitted"]


def test_fallback_backoff_increases_and_is_shared(tmp_path):
    now = [100.0]
    store = RouteThrottleStore(tmp_path / "state.sqlite", clock=lambda: now[0], jitter=lambda _a, _b: 1)
    first = store.acquire(ROUTE_A)
    store.finish(ROUTE_A, first["lease"], status_code=429)
    assert store.acquire(ROUTE_A)["retry_after"] == 2
    now[0] = 103
    second = store.acquire(ROUTE_A)
    store.finish(ROUTE_A, second["lease"], status_code=429)
    assert store.acquire(ROUTE_A)["retry_after"] == 4


def test_abandoned_leases_expire_and_heartbeats_survive_restart(tmp_path):
    now = [100.0]
    path = tmp_path / "state.sqlite"
    store = RouteThrottleStore(path, clock=lambda: now[0], max_concurrency=1, lease_seconds=10)
    acquired = store.acquire(ROUTE_A)
    now[0] = 108
    assert store.heartbeat(ROUTE_A, acquired["lease"])["renewed"]
    restarted = RouteThrottleStore(path, clock=lambda: now[0], max_concurrency=1, lease_seconds=10)
    now[0] = 111
    assert not restarted.acquire(ROUTE_A)["admitted"]
    now[0] = 119
    assert restarted.acquire(ROUTE_A)["admitted"]
    assert not restarted.heartbeat(ROUTE_A, acquired["lease"])["renewed"]


def test_lost_acquire_response_is_idempotent_and_does_not_leak_slots(server):
    request_id = "a" * 32
    first = client(server)._post("/acquire", {"request_id": request_id})
    second = client(server)._post("/acquire", {"request_id": request_id})
    assert first["lease"] == second["lease"]
    with client(server).acquire(timeout=0.2):
        pass
    client(server)._post("/finish", {"lease": first["lease"]})


def test_real_http_heartbeat_preserves_long_running_request(tmp_path):
    with RouteThrottleServer(tmp_path / "state.sqlite", TOKEN, max_concurrency=1,
                             lease_seconds=1, route_keys=[ROUTE_A]) as server:
        with client(server).acquire() as lease:
            time.sleep(1.2)
            lease.ensure_active()
            with pytest.raises(TimeoutError):
                client(server).acquire(timeout=0.1)
        with client(server).acquire(timeout=0.2):
            pass


def test_heartbeat_failure_closes_upstream_and_fails_closed(server, monkeypatch):
    instance = client(server)
    lease = instance.acquire()
    closed = threading.Event()
    lease.set_on_lost(closed.set)
    original = instance._post

    def fail_heartbeat(endpoint, payload, **kwargs):
        if endpoint == "/heartbeat":
            raise OSError("service unreachable")
        return original(endpoint, payload, **kwargs)

    monkeypatch.setattr(instance, "_post", fail_heartbeat)
    assert closed.wait(2)
    with pytest.raises(RuntimeError, match="heartbeat failed"):
        lease.ensure_active()
    lease.finish()


def test_authentication_registry_and_bounded_body(server):
    with pytest.raises(HTTPError) as error:
        RouteThrottleClient(server.url, "wrong-token", ROUTE_A).acquire(timeout=0.2)
    assert error.value.code == 401
    with pytest.raises(HTTPError) as error:
        client(server, "0" * 64).acquire(timeout=0.2)
    assert error.value.code == 403
    request = Request(server.url + "/finish", data=b"x" * 16385,
                      headers={"Authorization": "Bearer " + TOKEN})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400
    request = Request(server.url + "/health", headers={"Authorization": "Bearer " + TOKEN})
    with urlopen(request) as response:
        assert json.load(response) == {"ok": True}


def test_private_preview_header_and_filtered_response_headers(server, monkeypatch):
    instance = client(server, preview_token="preview-private")
    lease = instance.acquire()
    seen = []
    original = instance._post

    def record(endpoint, payload, **kwargs):
        seen.append((endpoint, payload))
        return original(endpoint, payload, **kwargs)

    monkeypatch.setattr(instance, "_post", record)
    lease.finish(status_code=200, headers={"Set-Cookie": "secret", "Retry-After": "1"})
    assert seen[0][1]["headers"] == {"Retry-After": "1"}


def test_finish_is_idempotent_and_context_releases_on_exception(server):
    lease = client(server).acquire()
    lease.finish(status_code=200)
    lease.finish(status_code=429, headers={"Retry-After": "100"})
    with pytest.raises(ValueError):
        with client(server).acquire(timeout=0.2):
            raise ValueError("upstream failed")
    with client(server).acquire(timeout=0.2):
        pass
