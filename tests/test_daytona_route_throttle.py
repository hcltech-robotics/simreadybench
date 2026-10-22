import json
import time
from types import SimpleNamespace

import pytest

from simready_bench import daytona_route_throttle as manager


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "_worker_network_preflight", lambda *args, **kwargs: None)
    monkeypatch.setenv("FIXTURE_ROUTE_KEY", "private-model-credential")
    policy = {
        "runtime": {"authority_state_dir": str(tmp_path)},
        "providers": {"hf": {
            "kind": "huggingface", "base_url": "https://router.huggingface.co/v1",
            "api_key_env": "FIXTURE_ROUTE_KEY",
            "project_budget": {"model": "fixture/model:route", "expires_at": time.time() + 3600},
        }},
    }
    manifest = {"models": [{"provider_ref": "hf", "model_id": "fixture/model:route"}]}
    return policy, manifest


class Client:
    def __init__(self):
        self.created = []
        self.deleted = False
        self.signed_calls = []
        self.sandbox = SimpleNamespace(
            id="governor-fixture", create_signed_preview_url=self.signed_preview,
        )

    def signed_preview(self, port, *, expires_in_seconds, **kwargs):
        self.signed_calls.append((port, expires_in_seconds))
        return SimpleNamespace(url="https://private-governor.daytona.test", token="private-signed-port-token")

    def create(self, params, **kwargs):
        self.created.append(params)
        return self.sandbox

    def get(self, identity):
        if self.deleted:
            from daytona.common.errors import DaytonaNotFoundError
            raise DaytonaNotFoundError("Deleted governor")
        assert identity == self.sandbox.id
        return self.sandbox

    def delete(self, sandbox, **kwargs):
        self.deleted = True


def test_one_private_governor_is_reused_and_credential_is_not_retained(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    starts, checks = [], []
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: starts.append(args))
    monkeypatch.setattr(manager, "_health", lambda descriptor: checks.append(descriptor))
    first = manager.ensure_route_throttle(policy, manifest, client, None, None)
    second = manager.ensure_route_throttle(policy, manifest, client, None, None)
    assert first == second
    assert len(client.created) == len(starts) == 1
    assert len(checks) == 2
    assert len(client.signed_calls) == 1
    assert client.signed_calls[0][0] == 8765
    assert 0 < client.signed_calls[0][1] <= 86400
    assert first["preview_token"] == ""
    assert client.created[0].public is False
    assert client.created[0].auto_stop_interval == 0
    receipts = list(tmp_path.glob("route-throttles/*/governor.json"))
    assert len(receipts) == 1
    raw = receipts[0].read_text()
    assert "private-model-credential" not in raw
    assert json.loads(raw)["sandbox_id"] == client.sandbox.id


def test_route_identity_is_shared_across_runs_and_separates_credentials(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    first = manager.route_identity(policy, manifest)
    policy["providers"]["hf"]["project_budget"]["expires_at"] += 60
    assert manager.route_identity(policy, manifest) == first
    monkeypatch.setenv("FIXTURE_ROUTE_KEY", "different-private-credential")
    assert manager.route_identity(policy, manifest) != first


def test_non_hf_route_does_not_require_governor(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    assert manager.requires_route_throttle(policy=policy, manifest=manifest)
    policy["providers"]["hf"].update(kind="openai", base_url="https://api.openai.com/v1")
    assert not manager.requires_route_throttle(policy=policy, manifest=manifest)


def test_unobserved_existing_governor_never_creates_replacement(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_health", lambda descriptor: None)
    manager.ensure_route_throttle(policy, manifest, client, None, None)
    def unavailable(identity):
        raise ConnectionError("Provider observation unavailable")
    client.get = unavailable
    with pytest.raises(ConnectionError):
        manager.ensure_route_throttle(policy, manifest, client, None, None)
    assert len(client.created) == 1


def test_different_authority_expiry_reuses_independent_governor(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_health", lambda descriptor: None)
    manager.ensure_route_throttle(policy, manifest, client, None, None)
    policy["providers"]["hf"]["project_budget"]["expires_at"] += 600
    manager.ensure_route_throttle(policy, manifest, client, None, None)
    assert len(client.created) == 1


def test_shared_governor_is_deleted_only_after_final_owner_releases(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_health", lambda descriptor: None)
    first_owner, second_owner = tmp_path / "first", tmp_path / "second"
    first = manager.ensure_route_throttle(policy, manifest, client, None, None, output_dir=first_owner)
    second = manager.ensure_route_throttle(policy, manifest, client, None, None, output_dir=second_owner)
    assert first == second
    assert manager.release_route_throttle(first, first_owner, client)["shared"]
    assert client.deleted is False
    assert manager.release_route_throttle(second, second_owner, client)["cleanup_confirmed"]
    assert client.deleted is True
    assert json.loads(open(first["receipt_path"]).read())["status"] == "deleted"


def test_unconfirmed_deletion_retains_receipt_and_blocks_replacement(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_health", lambda descriptor: None)
    descriptor = manager.ensure_route_throttle(policy, manifest, client, None, None)
    client.delete = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="deletion is unconfirmed"):
        manager.release_route_throttle(descriptor, tmp_path, client)
    with pytest.raises(RuntimeError, match="deletion is unresolved"):
        manager.ensure_route_throttle(policy, manifest, client, None, None)
    assert len(client.created) == 1


def test_failed_startup_deletes_its_unshared_governor(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    def failed(*args, **kwargs):
        raise RuntimeError("Runtime bootstrap failed")
    monkeypatch.setattr(manager, "_start", failed)
    with pytest.raises(RuntimeError, match="Runtime bootstrap failed"):
        manager.ensure_route_throttle(policy, manifest, client, None, None)
    assert client.deleted is True


def test_failed_worker_network_preflight_deletes_unshared_governor(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_health", lambda descriptor: None)
    def failed(*args, **kwargs):
        raise RuntimeError("shared_route_governor_worker_network_unreachable")
    monkeypatch.setattr(manager, "_worker_network_preflight", failed)
    with pytest.raises(RuntimeError, match="shared_route_governor_worker_network_unreachable"):
        manager.ensure_route_throttle(policy, manifest, client, None, None)
    assert client.deleted is True


def test_failed_worker_preflight_preserves_governor_with_another_owner(tmp_path, monkeypatch):
    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_health", lambda descriptor: None)
    descriptor = manager.ensure_route_throttle(policy, manifest, client, None, None, output_dir=tmp_path/'first')
    def failed(*args, **kwargs):
        raise RuntimeError("shared_route_governor_worker_network_unreachable")
    monkeypatch.setattr(manager, "_worker_network_preflight", failed)
    with pytest.raises(RuntimeError, match="shared_route_governor_worker_network_unreachable"):
        manager.ensure_route_throttle(policy, manifest, client, None, None, output_dir=tmp_path/'second')
    assert client.deleted is False
    assert len(json.loads(open(descriptor['receipt_path']).read())['owners']) == 1


def test_worker_network_preflight_keeps_capabilities_out_of_command_arguments(monkeypatch):
    uploads, commands = {}, []
    sandbox = SimpleNamespace(fs=SimpleNamespace(upload_file=lambda data,path: uploads.update({path:data})))
    def execute(sandbox, command, **kwargs):
        commands.append(command)
        return 'shared_route_governor_worker_network_ready' if 'docker exec' in command else ''
    monkeypatch.setattr('simready_bench.daytona._executor_command', execute)
    descriptor={'url':'https://signed-secret.daytona.test','token':'private-governor-token'}
    manager._worker_network_preflight(sandbox,descriptor)
    assert len(commands) == 2
    assert 'chmod 600' in commands[0]
    assert 'docker exec simready-route-throttle python -B' in commands[1]
    assert all(descriptor['url'] not in command and descriptor['token'] not in command for command in commands)
    assert json.loads(uploads['/run/simready-throttle/network-probe.json']) == descriptor
    script=uploads['/run/simready-throttle/network-probe.py'].decode()
    assert 'p.unlink()' in script and 'NoRedirect' in script


def test_worker_network_preflight_script_calls_actual_authenticated_http(tmp_path, monkeypatch):
    import contextlib
    import io
    from simready_bench.route_throttle import RouteThrottleServer

    uploads = {}
    sandbox = SimpleNamespace(fs=SimpleNamespace(upload_file=lambda data,path: uploads.update({path:data})))
    private_config = tmp_path/'probe.json'
    def execute(sandbox, command, **kwargs):
        if 'docker exec' not in command:
            return ''
        private_config.write_bytes(uploads['/run/simready-throttle/network-probe.json'])
        source = uploads['/run/simready-throttle/network-probe.py'].decode().replace(
            "'/state/network-probe.json'", repr(str(private_config))
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(source,'worker-network-probe','exec'),{})
        assert not private_config.exists()
        return output.getvalue().strip()
    monkeypatch.setattr('simready_bench.daytona._executor_command', execute)
    with RouteThrottleServer(tmp_path/'store.sqlite3', 'private-test-application-token',route_keys=['a'*64]) as server:
        manager._worker_network_preflight(sandbox,{'url':server.url,'token':'private-test-application-token'})


def test_simultaneous_coordinators_create_one_governor(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    policy, manifest = fixture(tmp_path, monkeypatch)
    client = Client()
    monkeypatch.setattr(manager, "_start", lambda *args, **kwargs: time.sleep(0.1))
    monkeypatch.setattr(manager, "_health", lambda descriptor: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(manager.ensure_route_throttle, policy, manifest, client, None, None) for _ in range(2)]
        first, second = [task.result(timeout=5) for task in tasks]
    assert first == second
    assert len(client.created) == 1


def test_actual_governor_health_authentication(tmp_path):
    from simready_bench.route_throttle import RouteThrottleServer

    with RouteThrottleServer(tmp_path / "route.sqlite3", "private-governor-app-token", route_keys=["a" * 64]) as server:
        descriptor = {
            "url": server.url, "token": "private-governor-app-token",
            "preview_token": "fixture-preview-token", "route_key": "a" * 64,
        }
        manager._health(descriptor)
        descriptor["token"] = "incorrect-private-token"
        with pytest.raises(Exception):
            manager._health(descriptor)


def test_start_uses_pinned_runtime_and_private_config(tmp_path, monkeypatch):
    from simready_bench.daytona import CPU_IMAGE

    uploads, sessions, commands = {}, [], []
    process = SimpleNamespace(
        create_session=lambda name: sessions.append(name),
        execute_session_command=lambda name, request, **kwargs: commands.append(request.command),
        exec=lambda command, **kwargs: SimpleNamespace(exit_code=0, result=""),
    )
    sandbox = SimpleNamespace(process=process, fs=SimpleNamespace(upload_file=lambda data, path: uploads.update({path: data})))
    receipt = {"token": "private-app-token", "expires_at": time.time() + 3600, "route_key": "a" * 64}
    manager._start(sandbox, receipt, cpu_image=CPU_IMAGE)
    config = json.loads(uploads["/run/simready-throttle/config.json"])
    assert config["host"] == "0.0.0.0" and config["port"] == 8765
    assert config["max_concurrency"] == 4 and config["route_keys"] == ["a" * 64]
    assert config["database"] == "/state/throttle.sqlite3"
    assert "--pull=never" in commands[-1] and CPU_IMAGE in commands[-1]
    assert "--config /state/config.json" in commands[-1]
