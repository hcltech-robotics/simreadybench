import copy
import io
import json
import os
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from simready_bench.daytona import (
    _archive_inputs,
    _check_wheel,
    _extract_inputs,
    _matrix_complete,
    _plan,
    _relay_command,
    build_wheel,
    remote_provider,
)
from simready_bench.provider_relay import Relay, RelayState


def experiment():
    provider = {
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "project_budget": {
            "project": "simready-bench",
            "authorization_id": "existing-parent",
            "approved": True,
            "upstream_origin": "https://api.openai.com",
            "model": "gpt-6.1-sol",
            "expires_at": time.time() + 3600,
            "maximum_requests": 3200,
            "maximum_spend_usd": 250,
            "pricing_status": "known",
            "external_spend_cap_verified": False,
        },
    }
    manifest = {
        "assets": [{"id": str(i)} for i in range(25)],
        "repetitions": 4,
        "models": [
            {
                "id": "selected",
                "model_id": "gpt-6.1-sol",
                "provider_ref": "selected",
                "adapter": "openai-responses",
            }
        ],
        "harnesses": [{"id": "pi"}],
        "skill_sets": [{"id": "none"}],
        "defaults": {"budgets": {"inference_requests": 32, "maximum_spend_usd": 2.5}},
    }
    policy = {
        "providers": {"selected": provider},
        "limits": {"maximum_spend_usd": 250, "maximum_matrix_runs": 100},
    }
    return manifest, policy


def test_full_matrix_reservation_and_remote_authority_preserve_parent():
    manifest, policy = experiment()
    original = copy.deepcopy(policy)
    provider, parent, requests, spend = _plan(manifest, policy)
    assert requests == 3200 and spend == 250
    remote = remote_provider(
        provider,
        port=12345,
        allocation_id="allocation",
        requests=requests,
        maximum_spend_usd=spend,
    )
    assert policy == original
    assert remote["project_budget"]["expires_at"] == parent.expires_at
    assert remote["project_budget"]["maximum_requests"] == parent.maximum_requests
    assert remote["project_budget"]["maximum_spend_usd"] == parent.maximum_spend_usd
    assert remote["project_budget"]["authorization_id"] != parent.authorization_id
    assert remote["project_budget"]["upstream_origin"] == "http://127.0.0.1:12345"
    assert remote["api_key_env"] == "SIMREADY_RELAY_KEY" and "model_spec" not in remote
    manifest["repetitions"] = 5
    with pytest.raises(ValueError, match="parent allowance"):
        _plan(manifest, policy)


def test_upstream_route_cannot_change_under_existing_authority():
    manifest, policy = experiment()
    policy["providers"]["selected"]["base_url"] = "https://example.com/v1"
    with pytest.raises(ValueError, match="endpoint differs"):
        _plan(manifest, policy)


@pytest.mark.parametrize("resources", [
    {}, {"cpus": True, "memory_mb": 4096}, {"cpus": 2, "memory_mb": 64},
    {"cpus": 2, "memory_mb": 4096, "extra": 1},
])
def test_invalid_actor_resources_are_rejected_before_spend_or_compute(tmp_path, monkeypatch, resources):
    from simready_bench import local
    from simready_bench.daytona import run_daytona

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid resources must not reserve an allowance")

    monkeypatch.setattr(local, "ParentAuthorityLedger", unexpected)
    manifest, policy = experiment()
    policy["runtime"] = {"container_resources": resources}
    with pytest.raises(ValueError, match="Actor resources"):
        run_daytona(
            manifest, policy=policy, output_dir=tmp_path,
            assets_root=tmp_path / "assets", skills_root=tmp_path / "skills",
        )
    assert not (tmp_path / "daytona-delivery.json").exists()


def test_collection_retains_attempts_when_aggregate_report_is_missing(tmp_path):
    from simready_bench.daytona import _collect_portable

    output = tmp_path / "collected"
    portable = output / "daytona-executor/output"
    attempt = portable / "attempts/000-asset"
    attempt.mkdir(parents=True)
    (attempt / "results.json").write_text('{"results":[]}')
    (attempt / "candidate.usda").write_text("#usda 1.0\n")
    (portable / "experiment.json").write_text("remote configuration")
    assert not _collect_portable(portable, output)
    assert (output / "attempts/000-asset/candidate.usda").read_text() == "#usda 1.0\n"
    assert (portable / "experiment.json").is_file()


def test_canonical_completion_requires_all_four_repeats_of_every_reference():
    manifest, _ = experiment()
    manifest["repetitions"] = 4
    rows = [
        {
            "asset_id": str(index),
            "repetition": repetition,
            "status": "completed",
            "simready_reward": 80,
        }
        for index in range(25)
        for repetition in range(4)
    ]
    result = {"planned_conditions": 100, "completed_conditions": 100, "results": rows}
    assert _matrix_complete(result, manifest)
    result["results"] = rows[:25]
    assert not _matrix_complete(result, manifest)
    result["results"] = copy.deepcopy(rows)
    result["results"][0]["status"] = "blocked"
    assert not _matrix_complete(result, manifest)
    result["results"] = copy.deepcopy(rows)
    result["results"][0]["asset_id"] = "24"
    assert not _matrix_complete(result, manifest)
    result["results"] = copy.deepcopy(rows)
    result["results"][1]["repetition"] = 0
    assert not _matrix_complete(result, manifest)


@pytest.mark.parametrize("score", [None, float("nan"), True, -1, 101])
def test_unmeasured_or_invalid_scores_cannot_complete_a_run(score):
    manifest, _ = experiment()
    rows = [
        {
            "asset_id": str(index),
            "repetition": repetition,
            "status": "completed",
            "simready_reward": 80,
        }
        for index in range(25)
        for repetition in range(4)
    ]
    rows[0]["simready_reward"] = score
    assert not _matrix_complete(
        {"planned_conditions": 100, "completed_conditions": 100, "results": rows},
        manifest,
    )


def test_relay_sidecar_uses_existing_python_and_only_private_mount():
    argv = _relay_command()
    text = " ".join(argv)
    assert "--network host" in text and "--read-only" in argv
    assert "SSL_CERT_FILE=/run/host-ca.pem" in argv
    assert "HTTP_PROXY" in argv and "HTTPS_PROXY" in argv
    assert "src=/etc/ssl/cert.pem,dst=/run/host-ca.pem,readonly" in text
    assert "src=/run/simready-benchmark,dst=/run/simready-benchmark" in text
    assert "src=/work" not in text


@pytest.mark.skipif(sys.platform == "win32", reason="The executor bootstrap runs in a Linux shell")
@pytest.mark.parametrize("running,public_ready", [
    ("dockerd", True), ("dockerd", False), ("wrapper", False), ("none", False),
])
def test_executor_bootstrap_preserves_an_active_or_loading_public_runtime(tmp_path, monkeypatch, running, public_ready):
    import shlex
    import subprocess
    from simready_bench.daytona import _bootstrap_command

    executables = tmp_path / "bin"
    executables.mkdir()
    pidof = executables / "pidof"
    pidof.write_text(
        '#!/bin/sh\n'
        'case "$1:$RUNNING" in dockerd:dockerd|dockerd-entrypoint.sh:wrapper) exit 0;; esac\n'
        'exit 1\n'
    )
    entrypoint = executables / "dockerd-entrypoint.sh"
    entrypoint.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$1" >> "$STARTED"\n'
        'rm -f "$PRIVATE_SOCKET" "$PUBLIC_SOCKET"\n'
    )
    pidof.chmod(0o755)
    entrypoint.chmod(0o755)
    started = tmp_path / "started"
    private_socket = tmp_path / "private.sock"
    public_socket = tmp_path / "public.sock"
    if running != "none":
        private_socket.write_text("existing private socket")
    if public_ready:
        public_socket.write_text("existing public socket")
    monkeypatch.setenv("PATH", str(executables) + ":" + os.environ["PATH"])
    for name, value in {
        "RUNNING": running, "STARTED": started,
        "PRIVATE_SOCKET": private_socket, "PUBLIC_SOCKET": public_socket,
    }.items():
        monkeypatch.setenv(name, str(value))
    command = shlex.split(_bootstrap_command(tmp_path / "dockerd.log"))
    command[-1] += "; wait"
    subprocess.run(command, check=True, timeout=5)
    if running == "none":
        assert started.read_text() == "dockerd\n"
    else:
        assert not started.exists()
        assert private_socket.read_text() == "existing private socket"
        assert public_socket.exists() is public_ready


def test_staged_usd_composition_and_skills_round_trip_without_links(tmp_path):
    roots = {
        name: tmp_path / name
        for name in ("assets", "skills", "input_views", "private_manifests")
    }
    for root in roots.values():
        root.mkdir()
    (roots["assets"] / "object.usda").write_text("#usda 1.0\n")
    (roots["assets"] / "Textures").mkdir()
    (roots["assets"] / "Textures" / "colour.png").write_bytes(b"texture")
    (roots["skills"] / "SKILL.md").write_text("Author USD")
    archive = tmp_path / "inputs.tar.gz"
    _archive_inputs(roots, archive)
    output = tmp_path / "remote"
    _extract_inputs(archive, output)
    assert (output / "assets" / "Textures" / "colour.png").read_bytes() == b"texture"
    assert (output / "skills" / "SKILL.md").read_text() == "Author USD"


@pytest.mark.parametrize(
    "name,kind",
    [("../outside", None), ("assets/link", tarfile.SYMTYPE), ("operator.json", None)],
)
def test_input_archive_rejects_escape_links_and_private_root(tmp_path, name, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        if kind:
            member.type = kind
            member.linkname = "/etc/passwd"
            stream.addfile(member)
        else:
            member.size = 1
            stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Invalid staged"):
        _extract_inputs(archive, tmp_path / "remote")
    assert not (tmp_path / "outside").exists()


def test_installed_wheel_contains_complete_executor_and_no_bytecode(tmp_path):
    wheel = build_wheel(tmp_path)
    assert _check_wheel(wheel) == wheel
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "simready_bench/daytona.py" in names
        assert "simready_bench/provider_relay.py" in names
        assert "simready_bench/engine/inference_gateway.py" in names
        assert len(names) == len(set(names))
        assert all(
            name.startswith("simready_bench/") or ".dist-info/" in name
            for name in names
        )
        assert not any(name.endswith(".pyc") or "__pycache__" in name for name in names)
        metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
        assert b"Name: simready-bench" in archive.read(metadata)
        assert b"Requires-Dist: harbor" in archive.read(metadata)


def test_stale_distribution_cannot_mislabel_the_executor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from simready_bench import daytona

    monkeypatch.setattr(
        daytona.metadata, "distribution", lambda name: SimpleNamespace(version="0.0.0")
    )
    with pytest.raises(ValueError, match="Reinstall SimReadyBench"):
        build_wheel(tmp_path)


class UpstreamResponse(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}


class RecordingOpener:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout):
        from types import SimpleNamespace

        class ClosedSocket:
            def fileno(self):
                return -1

            def settimeout(self, value):
                raise OSError("The completed HTTP connection is closed")

        self.requests.append(request)
        response = UpstreamResponse(
            b'{"object":"response.input_tokens","input_tokens":25}'
        )
        response.fp = SimpleNamespace(raw=SimpleNamespace(_sock=ClosedSocket()))
        return response


def test_relay_forwards_count_and_generation_exactly_without_exposing_key(tmp_path):
    import http.server

    upstream = RecordingOpener()
    config = {
        "model": "gpt-6.1-sol",
        "marker": "test-marker",
        "api_key": "private-upstream-key",
        "base_url": "https://api.openai.com/v1",
        "maximum_requests": 1,
        "expires_at": time.time() + 60,
    }
    state = RelayState(config, directory=tmp_path, opener=upstream)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(route, body, token="test-marker"):
        request = urllib.request.Request(
            "http://127.0.0.1:" + str(server.server_port) + route,
            data=body,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
        )
        return client.open(request, timeout=5)

    raw = b'{"model":"gpt-6.1-sol","input":"Reconstruct the asset","stream":false}'
    try:
        with post("/v1/responses/input_tokens", raw) as response:
            assert json.load(response)["input_tokens"] == 25
        with post("/v1/responses", raw) as response:
            assert response.status == 200
        assert state.counters == 1 and state.requests == 1
        assert [request.full_url for request in upstream.requests] == [
            "https://api.openai.com/v1/responses/input_tokens",
            "https://api.openai.com/v1/responses",
        ]
        assert all(request.data == raw for request in upstream.requests)
        assert all(
            request.get_header("Authorization") == "Bearer private-upstream-key"
            for request in upstream.requests
        )
        with pytest.raises(urllib.error.HTTPError) as exhausted:
            post("/v1/responses", raw)
        assert exhausted.value.code == 400 and len(upstream.requests) == 2
        with pytest.raises(urllib.error.HTTPError) as unauthorised:
            post("/v1/responses", raw, token="wrong")
        assert unauthorised.value.code == 403
        assert "private-upstream-key" not in (tmp_path / "ledger.json").read_text()
        assert "test-marker" not in (tmp_path / "ledger.json").read_text()
        assert all(event["status"] == "completed" for event in state.events)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "observation_unknown,publication_failure",
    [(False, False), (True, False), (False, True)],
)
@pytest.mark.parametrize("actor_resources", [None, {"cpus": 3, "memory_mb": 8192}])
def test_full_executor_reserves_before_create_exports_then_deletes(
    tmp_path, monkeypatch, observation_unknown, publication_failure, actor_resources
):
    import daytona
    from types import SimpleNamespace
    from daytona.common.errors import DaytonaNotFoundError
    from simready_bench import local
    from simready_bench.daytona import run_daytona

    manifest, policy = experiment()
    output = tmp_path / "run"
    output.mkdir()
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (output / name).mkdir()
    (output / "experiment.json").write_text("original launcher manifest")
    policy["runtime"] = {"authority_state_dir": str(tmp_path / "authority")}
    if actor_resources is not None:
        policy["runtime"]["container_resources"] = actor_resources
    events = []
    uploads = {}
    commands = []
    archive_path = tmp_path / "remote.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, value in {
            "work/output/report.html": b"<h1>Results</h1>",
            "work/output/results.json": b'{"results":[]}',
            "work/output/scores.csv": b"asset,score\n",
            "work/output/experiment.json": b"remote manifest",
            "work/run-result.json": json.dumps(
                {
                    "planned_conditions": 100,
                    "completed_conditions": 100,
                    "results": [
                        {
                            "asset_id": str(index),
                            "repetition": repetition,
                            "status": "completed",
                            "simready_reward": 80,
                        }
                        for index in range(25)
                        for repetition in range(4)
                    ],
                }
            ).encode(),
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))

    class Ledger:
        def __init__(self, state):
            pass

        def reserve(self, parent, **kwargs):
            events.append(("reserve", kwargs["requests"], kwargs["maximum_spend_usd"]))
            return {"status": "reserved", "allocation_id": kwargs["allocation_id"]}

    class FileSystem:
        def upload_file(self, source, target):
            uploads[target] = (
                source if isinstance(source, bytes) else Path(source).read_bytes()
            )

        def download_file(self, source, target):
            assert source == "/tmp/simready-benchmark-results.tar.gz"
            Path(target).write_bytes(archive_path.read_bytes())

    class Process:
        def create_session(self, name):
            pass

        def execute_session_command(self, session, request, timeout):
            commands.append(request.command)
            return SimpleNamespace(cmd_id="test-command")

        def get_session_command(self, *args, **kwargs):
            if observation_unknown:
                raise TimeoutError("Session observation unavailable")
            return SimpleNamespace(exit_code=0)

        def exec(self, command, timeout):
            commands.append(command)
            value = ""
            if "ready.json" in command:
                value = '{"port":32000}'
            if "stat -c %g" in command:
                value = "1000"
            return SimpleNamespace(exit_code=0, result=value)

    sandbox = SimpleNamespace(id="owned-executor", process=Process(), fs=FileSystem())

    class Client:
        def __init__(self, config):
            pass

        def create(self, params, timeout):
            assert 50 <= params.ttl_minutes <= 60
            events.append(("create",))
            return sandbox

        def delete(self, selected, timeout, wait):
            events.append(("delete", selected.id))

        def get(self, identity):
            raise DaytonaNotFoundError("deleted")

    monkeypatch.setattr(daytona, "Daytona", Client)
    monkeypatch.setattr(local, "ParentAuthorityLedger", Ledger)
    monkeypatch.setenv("DAYTONA_API_KEY", "private-daytona-key")
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-key")
    monkeypatch.setattr("simready_bench.daytona.time.sleep", lambda seconds: None)
    wheel = build_wheel(tmp_path / "wheel")
    if observation_unknown:
        from simready_bench.daytona import DaytonaObservationError

        with pytest.raises(DaytonaObservationError):
            run_daytona(
                manifest,
                policy=policy,
                output_dir=output,
                assets_root=output / "assets",
                skills_root=output / "skills",
                wheel=wheel,
            )
        assert events == [("reserve", 3200, 250), ("create",)]
        delivery = json.loads((output / "daytona-delivery.json").read_text())
        assert (
            delivery["status"] == "unobserved"
            and delivery["cleanup_confirmed"] is False
        )
        assert delivery["parent_allocation"]["status"] == "reserved"
        assert delivery.get("collection") != "completed"
        return
    publications = []

    def publish(report, *, final=False):
        assert events[-1] == ("delete", "owned-executor")
        publications.append((report, final))
        if publication_failure:
            raise ConnectionError("repository unavailable")

    options = {
        "policy": policy,
        "output_dir": output,
        "assets_root": output / "assets",
        "skills_root": output / "skills",
        "wheel": wheel,
        "on_report": publish,
    }
    if publication_failure:
        from simready_bench.daytona import DaytonaReportError

        with pytest.raises(DaytonaReportError) as captured:
            run_daytona(manifest, **options)
        result = captured.value.result
        saved = json.loads((output / "daytona-delivery.json").read_text())
        assert saved["status"] == "completed" and saved["cleanup_confirmed"] is True
        assert saved["collection"] == "completed"
        assert saved["report_callback_error"] == "ConnectionError"
    else:
        result = run_daytona(manifest, **options)
    assert len(publications) == 1 and publications[0][1] is True
    assert events == [("reserve", 3200, 250), ("create",), ("delete", "owned-executor")]
    assert result["completed_conditions"] == 100
    assert result["deployment"]["cleanup_confirmed"] is True
    assert (output / "report.html").is_file()
    assert (output / "experiment.json").read_text() == "original launcher manifest"
    assert b"private-openai-key" in uploads["/run/simready-benchmark/provider.json"]
    assert all(
        b"private-openai-key" not in value
        for name, value in uploads.items()
        if name.startswith("/work/")
    )
    assert not any(
        "private-openai-key" in value or "private-daytona-key" in value
        for value in commands
    )
    assert "private-openai-key" not in (output / "daytona-delivery.json").read_text()
    envelope = json.loads(uploads["/work/controller.json"])
    assert envelope["container_resources"] == (
        actor_resources or {"cpus": 2, "memory_mb": 4096}
    )
    assert result["deployment"]["container_resources"] == envelope["container_resources"]


def test_custom_runtime_mapping_reaches_relay_controller_and_actor(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import _relay_command

    cpu_image = "sha256:" + "1" * 64
    assert cpu_image in _relay_command(cpu_image)


def test_result_archive_rejects_escape_and_links(tmp_path):
    from simready_bench.daytona import _extract_results

    for name, kind in [("../escape", None), ("work/link", tarfile.SYMTYPE)]:
        archive = tmp_path / ("bad-" + str(kind) + ".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            member = tarfile.TarInfo(name)
            if kind:
                member.type = kind
                member.linkname = "/etc/passwd"
                stream.addfile(member)
            else:
                member.size = 1
                stream.addfile(member, io.BytesIO(b"x"))
        with pytest.raises(ValueError):
            _extract_results(archive, tmp_path / "results")
    assert not (tmp_path / "escape").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Controller policies require Linux")
@pytest.mark.parametrize("actor_resources", [None, {"cpus": 4, "memory_mb": 6144}])
def test_remote_controller_builds_a_valid_current_policy(tmp_path, monkeypatch, actor_resources):
    from simready_bench import benchmark, daytona
    from simready_bench.config import validate_policy
    from simready_bench.runtime_images import DEFAULT_IMAGES

    root = tmp_path / "work"
    root.mkdir()
    with tarfile.open(root / "inputs.tar.gz", "w:gz"):
        pass
    manifest, policy = experiment()
    provider = remote_provider(
        policy["providers"]["selected"],
        port=12345,
        allocation_id="test",
        requests=3200,
        maximum_spend_usd=250,
    )
    envelope = {
        "manifest": manifest,
        "provider": provider,
        "provider_ref": "selected",
        "marker": "test-marker",
        "maximum_spend_usd": 250,
        "limits": policy["limits"],
        "actor_image": DEFAULT_IMAGES["actor"],
        "verifier_image": DEFAULT_IMAGES["verifier"],
        "cpu_image": DEFAULT_IMAGES["cpu"],
        "work_dir": str(tmp_path / "trials"),
    }
    if actor_resources is not None:
        envelope["container_resources"] = actor_resources
    request = tmp_path / "controller.json"
    request.write_text(json.dumps(envelope))
    monkeypatch.setenv("SIMREADY_RELAY_KEY", "previous-marker")
    monkeypatch.setattr(
        daytona, "Path", lambda value: root if str(value) == "/work" else Path(value)
    )

    def execute(manifest, *, policy, output_dir, work_dir, on_progress):
        validate_policy(policy, base_dir=root)
        assert "actor_backend" not in policy["runtime"]
        assert policy["runtime"]["container_resources"] == (
            actor_resources or {"cpus": 2, "memory_mb": 4096}
        )
        assert callable(on_progress)
        return {
            "planned_conditions": 100,
            "completed_conditions": 100,
            "results": [
                {
                    "asset_id": str(index),
                    "repetition": repetition,
                    "status": "completed",
                    "simready_reward": 80,
                }
                for index in range(25)
                for repetition in range(4)
            ],
        }

    monkeypatch.setattr(benchmark, "execute_local_sequential", execute)
    assert daytona.execute_remote(request) == 0
    assert (
        json.loads((root / "controller-status.json").read_text())["status"]
        == "completed"
    )
