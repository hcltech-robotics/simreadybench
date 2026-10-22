"""Private, persistent route governor shared by remote benchmark executors."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
from pathlib import Path
import secrets
import shlex
import time
import urllib.request
from uuid import uuid4


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


@contextmanager
def _route_lock(directory):
    from .parallel import coordinator_lock

    deadline = time.monotonic() + 900
    while True:
        manager = coordinator_lock(directory)
        try:
            manager.__enter__()
            break
        except ValueError as error:
            if "Another benchmark coordinator" not in str(error) or time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


def route_identity(policy, manifest):
    model = manifest["models"][0]
    provider = policy["providers"][model["provider_ref"]]
    credential = os.environ.get(provider.get("api_key_env", ""))
    if not credential:
        raise ValueError("The selected provider credential is unavailable")
    identity = {
        "origin": provider["base_url"].rstrip("/"),
        "provider": provider["kind"],
        "model": model.get("model_id", provider["project_budget"]["model"]),
        "credential_sha256": hashlib.sha256(credential.encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def requires_route_throttle(*, policy, manifest):
    model = manifest["models"][0]
    provider = policy["providers"][model["provider_ref"]]
    return (
        provider.get("kind") == "huggingface"
        or provider.get("base_url", "").rstrip("/") == "https://router.huggingface.co/v1"
        or policy.get("runtime", {}).get("shared_route_throttle") is True
    )


def _health(descriptor):
    headers = {"Authorization": "Bearer " + descriptor["token"]}
    if descriptor.get("preview_token"):
        headers["X-Daytona-Preview-Token"] = descriptor["preview_token"]
    request = urllib.request.Request(
        descriptor["url"].rstrip("/") + "/health",
        headers=headers,
    )
    # Never forward the preview capability or application token through redirects.
    from .provider_relay import NoRedirect

    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=10) as response:
        body = response.read(65537)
        if response.status != 200 or len(body) > 65536:
            raise RuntimeError("The shared route governor health response is invalid")
        value = json.loads(body)
        if not isinstance(value, dict) or value.get("ok") is not True:
            raise RuntimeError("The shared route governor is unavailable")


def _start(sandbox, receipt, *, cpu_image):
    from daytona import SessionExecuteRequest
    from .daytona import _bootstrap_command, _executor_command
    from . import route_throttle

    def command(text, timeout=30):
        return _executor_command(sandbox, text, clean=lambda value: "Governor command failed", timeout=timeout)

    sandbox.process.create_session("simready-throttle-dockerd")
    sandbox.process.execute_session_command(
        "simready-throttle-dockerd",
        SessionExecuteRequest(command=_bootstrap_command(), run_async=True),
        timeout=10,
    )
    command("timeout 510 sh -c 'until docker info >/dev/null 2>&1; do sleep 2; done'", timeout=520)
    command("mkdir -p /run/simready-throttle; chmod 700 /run/simready-throttle")
    sandbox.fs.upload_file(str(Path(route_throttle.__file__)), "/run/simready-throttle/route_throttle.py")
    config = {
        "database": "/state/throttle.sqlite3", "host": "0.0.0.0", "port": 8765,
        "token": receipt["token"], "max_concurrency": 4, "lease_seconds": 120,
        "expires_at": receipt["expires_at"], "route_keys": [receipt["route_key"]],
    }
    sandbox.fs.upload_file(json.dumps(config).encode(), "/run/simready-throttle/config.json")
    command("chmod 600 /run/simready-throttle/*")
    argv = [
        "docker", "run", "--rm", "--pull=never", "--name", "simready-route-throttle",
        "--network", "host", "--user", "0:0", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--cpus", "1", "--memory", "256m",
        "--mount", "type=bind,src=/run/simready-throttle,dst=/state",
        "--mount", "type=bind,src=/etc/ssl/cert.pem,dst=/run/host-ca.pem,readonly",
        "--env", "SSL_CERT_FILE=/run/host-ca.pem",
    ]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        argv += ["--env", name]
    argv += [
        "--entrypoint", "python", cpu_image,
        "-B", "/state/route_throttle.py", "--config", "/state/config.json",
    ]
    sandbox.process.create_session("simready-route-throttle")
    sandbox.process.execute_session_command(
        "simready-route-throttle", SessionExecuteRequest(command=shlex.join(argv), run_async=True), timeout=10,
    )


def _worker_network_preflight(sandbox, descriptor):
    """Verify the signed HTTPS endpoint from the pinned governor container."""
    from .daytona import _executor_command

    probe = '''import json,pathlib,sys,urllib.request
p=pathlib.Path('/state/network-probe.json')
try:
    raw=p.read_text()
    p.unlink()
    config=json.loads(raw)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):
            return None
    request=urllib.request.Request(config['url'].rstrip('/')+'/health',headers={'Authorization':'Bearer '+config['token']})
    with urllib.request.build_opener(NoRedirect()).open(request,timeout=15) as response:
        raw=response.read(65537)
        if response.status!=200 or len(raw)>65536 or json.loads(raw).get('ok') is not True:
            raise ValueError('Invalid health response')
except Exception:
    print('shared_route_governor_worker_network_unreachable')
    sys.exit(1)
print('shared_route_governor_worker_network_ready')
'''
    sandbox.fs.upload_file(probe.encode(), "/run/simready-throttle/network-probe.py")
    sandbox.fs.upload_file(json.dumps({key: descriptor[key] for key in ("url", "token")}).encode(), "/run/simready-throttle/network-probe.json")
    clean = lambda value: "shared_route_governor_worker_network_unreachable"
    try:
        _executor_command(sandbox, "chmod 600 /run/simready-throttle/network-probe.*", clean=clean)
        result = _executor_command(
            sandbox,
            "docker exec simready-route-throttle python -B /state/network-probe.py",
            clean=clean, timeout=25,
        )
        if result != "shared_route_governor_worker_network_ready":
            raise RuntimeError("shared_route_governor_worker_network_unreachable")
    except Exception:
        raise RuntimeError("shared_route_governor_worker_network_unreachable") from None


def ensure_route_throttle(policy, manifest, client, cpu_snapshot, runtime_images, *, output_dir=None):
    """Reconnect the route governor or create one under a host-wide route lock."""
    from daytona import CreateSandboxFromSnapshotParams
    from daytona.common.errors import DaytonaNotFoundError
    from .daytona import CPU_IMAGE, CPU_SNAPSHOT

    route_key = route_identity(policy, manifest)
    cpu_image = (runtime_images or {}).get("cpu", CPU_IMAGE)
    if not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", cpu_image):
        raise ValueError("The route governor requires a pinned CPU image digest")
    root = Path(os.path.abspath(policy["runtime"]["authority_state_dir"])) / "route-throttles"
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory = root / route_key
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.resolve() != directory:
        raise ValueError("The route governor state must remain in its authority directory")
    path = directory / "governor.json"
    expires_at = policy["providers"][manifest["models"][0]["provider_ref"]]["project_budget"]["expires_at"]
    owner = str(Path(os.path.abspath(output_dir or policy["runtime"]["authority_state_dir"])))
    with _route_lock(directory):
        receipt, sandbox = None, None
        if path.exists():
            if path.is_symlink() or path.stat().st_nlink != 1:
                raise ValueError("A regular private route governor receipt is required")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("route_key") != route_key:
                raise ValueError("The saved governor route identity changed")
            try:
                sandbox = client.get(receipt.get("sandbox_id") or receipt["sandbox_name"])
            except DaytonaNotFoundError:
                # Only a positive provider absence allows a replacement.
                history = directory / ("retired-" + uuid4().hex + ".json")
                _write(history, {**receipt, "provider_absent_at": time.time()})
                receipt = None
            if receipt is not None and receipt.get("status") in {"deleting", "deleted"}:
                raise RuntimeError("The previous route governor deletion is unresolved")
        if sandbox is None:
            if not math.isfinite(expires_at) or expires_at <= time.time() + 600:
                raise ValueError("The route governor requires an unexpired run authority")
            receipt = {
                "schema": "simready-route-governor-v1", "route_key": route_key,
                "sandbox_name": "simready-route-" + uuid4().hex[:20],
                "token": secrets.token_urlsafe(32), "expires_at": time.time() + 86400,
                "status": "creating", "created_at": time.time(),
                "owners": {},
            }
            _write(path, receipt)
            sandbox = client.create(CreateSandboxFromSnapshotParams(
                name=receipt["sandbox_name"], snapshot=cpu_snapshot or CPU_SNAPSHOT,
                public=False, ttl_minutes=math.ceil((receipt["expires_at"] - time.time()) / 60),
                auto_stop_interval=0, auto_delete_interval=1, ephemeral=False,
                labels={"project": "simreadybench", "role": "benchmark-inference-governor", "route": route_key},
            ), timeout=90)
            receipt.update(sandbox_id=sandbox.id, status="created")
            _write(path, receipt)
        if receipt.get("expires_at", 0) < time.time() + 2400:
            raise RuntimeError("The existing route governor has insufficient service lifetime")
        if receipt["status"] in {"creating", "created"}:
            # A crashed startup must be contained rather than starting a second server.
            if receipt["status"] == "creating":
                receipt.update(sandbox_id=sandbox.id, status="created")
            receipt["status"] = "starting"
            _write(path, receipt)
            try:
                _start(sandbox, receipt, cpu_image=cpu_image)
            except BaseException:
                if not receipt.get("owners"):
                    release_route_throttle({"receipt_path": str(path), "route_key": route_key, "token": receipt["token"]}, owner, client)
                raise
        try:
            if not receipt.get("signed_url") or receipt.get("signed_expires_at", 0) <= time.time() + 2400:
                issued_at = time.time()
                lifetime = min(86400, int(receipt["expires_at"] - issued_at) - 30)
                signed = sandbox.create_signed_preview_url(8765, expires_in_seconds=lifetime, request_timeout=30)
                receipt.update(signed_url=signed.url.rstrip("/"), signed_token=signed.token,
                               signed_expires_at=issued_at + lifetime)
                _write(path, receipt)
            from urllib.parse import urlsplit

            parsed = urlsplit(receipt["signed_url"])
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("A private HTTPS governor preview URL is required")
        except BaseException:
            if not receipt.get("owners"):
                release_route_throttle({"receipt_path": str(path), "route_key": route_key, "token": receipt["token"]}, owner, client)
            raise
        descriptor = {"url": receipt["signed_url"], "token": receipt["token"], "preview_token": "", "route_key": route_key, "receipt_path": str(path)}
        deadline = time.monotonic() + 60
        while True:
            try:
                _health(descriptor)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    if not receipt.get("owners"):
                        release_route_throttle(descriptor, owner, client)
                    raise RuntimeError("The shared route governor is unreachable") from None
                time.sleep(1)
        try:
            _worker_network_preflight(sandbox, descriptor)
        except Exception:
            if not receipt.get("owners"):
                release_route_throttle(descriptor, owner, client)
            raise
        receipt.update(status="ready", url=descriptor["url"], preview_token="", last_verified_at=time.time())
        receipt.setdefault("owners", {})[owner] = expires_at
        _write(path, receipt)
        return descriptor


def release_route_throttle(descriptor, owner, client):
    """Release a stopped coordinator and delete only its unshared governor."""
    from daytona.common.errors import DaytonaNotFoundError

    path = Path(descriptor["receipt_path"])
    if path.resolve() != path or path.is_symlink() or path.name != "governor.json":
        raise ValueError("The private governor receipt path is invalid")
    with _route_lock(path.parent):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["route_key"] != descriptor["route_key"] or receipt["token"] != descriptor["token"]:
            raise ValueError("The governor release identity changed")
        owners = dict(receipt.get("owners", {}))
        owners.pop(str(Path(os.path.abspath(owner))), None)
        receipt["owners"] = owners
        if owners:
            _write(path, receipt)
            return {"shared": True, "cleanup_confirmed": False}
        receipt["status"] = "deleting"
        _write(path, receipt)
        try:
            sandbox = client.get(receipt["sandbox_id"])
        except DaytonaNotFoundError:
            sandbox = None
        if sandbox is not None:
            client.delete(sandbox, timeout=60, wait=True)
            try:
                client.get(receipt["sandbox_id"])
            except DaytonaNotFoundError:
                pass
            else:
                raise RuntimeError("The route governor deletion is unconfirmed")
        receipt.update(status="deleted", cleanup_confirmed=True, deleted_at=time.time())
        _write(path, receipt)
        return {"shared": False, "cleanup_confirmed": True}
