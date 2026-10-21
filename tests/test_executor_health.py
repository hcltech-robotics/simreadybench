"""Bound detached or stalled cloud workers without replacing live ownership."""
import json
from types import SimpleNamespace

import pytest

from simready_bench import daytona


def monitor_fixture(monkeypatch, tmp_path, *, stale=True, running=True, exit_code=0,
                    relay_events=()):
    clock = [10000.0]
    record = {
        "sandbox_id": "saved-sandbox", "run_id": "saved-run",
        "command_id": "saved-command", "session_name": "saved-session",
        "deadline_at": 20000, "last_activity_at": 5000 if stale else 10000,
        "activity_counters": [1, 0],
        "health_policy": daytona._health_policy({}),
    }
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        if "docker inspect" in command:
            return SimpleNamespace(exit_code=0, result=json.dumps({"Running": running, "Status": "running" if running else "exited", "ExitCode": exit_code}))
        return SimpleNamespace(exit_code=0, result="simready-containers-stopped")

    def download(path, *args):
        if path == "/work/controller-status.json":
            return b'{"status":"running"}'
        if path == "/run/simready-benchmark/ledger.json":
            return json.dumps({"generation_requests": 1, "input_counter_requests": 0,
                               "events": list(relay_events)}).encode()
        raise FileNotFoundError(path)

    sandbox = SimpleNamespace(
        process=SimpleNamespace(exec=execute, get_session_command=lambda *a, **k: SimpleNamespace(exit_code=None)),
        fs=SimpleNamespace(download_file=download),
    )
    monkeypatch.setattr(daytona.time, "time", lambda: clock[0])
    monkeypatch.setattr(daytona.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(daytona.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr("simready_bench.controller_runner.forward_telemetry", lambda *a, **k: 0)
    return sandbox, record, commands, clock


def test_stalled_running_controller_is_stopped_before_collection(monkeypatch, tmp_path):
    sandbox, record, commands, _ = monitor_fixture(monkeypatch, tmp_path)
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["terminal_confirmed"] is True
    assert record["termination_reason"] == "executor_stalled"
    assert record["controller_exit"] == 137
    assert any("docker kill" in command for command in commands)
    proof = json.loads((tmp_path / "executor-terminal-proof.json").read_text())
    assert proof["all_containers_stopped"] is True
    assert proof["sandbox_id"] == record["sandbox_id"]


def test_open_circuit_stops_the_owned_worker_before_further_observation(monkeypatch, tmp_path):
    sandbox, record, commands, _ = monitor_fixture(monkeypatch, tmp_path, stale=False)
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None,
        stop_requested=lambda: True)
    assert record['termination_reason'] == 'inference_transport_circuit_open'
    assert record['terminal_confirmed'] is True
    assert any('docker kill' in command for command in commands)


def test_truncated_provider_stream_stops_live_worker_for_native_collection(monkeypatch, tmp_path):
    failure = {
        "request_ordinal": 1, "condition_request_ordinal": 1,
        "path": "/chat/completions", "status": "transport_error",
        "http_status": 200, "error_type": "IncompleteRead",
        "started_at": 9990, "finished_at": 9999,
    }
    sandbox, record, commands, _ = monitor_fixture(
        monkeypatch, tmp_path, stale=False, relay_events=[failure],
    )
    record["condition_retry_counts"] = {"saved-condition": 2}
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["termination_reason"] == "inference_transport_circuit_open"
    assert record["terminal_confirmed"] is True
    assert record["provider_transport_failure"]["error_type"] == "IncompleteRead"
    assert len(record["provider_transport_failure"]["ledger_sha256"]) == 64
    assert record["condition_retry_counts"] == {"saved-condition": 2}
    assert record["activity_counters"] == [1, 0]
    assert any("docker kill" in command for command in commands)
    assert (tmp_path / "executor-terminal-proof.json").is_file()


@pytest.mark.parametrize("status", ["admitted", "completed", "retryable_rejection", "client_disconnected"])
def test_nonfatal_relay_event_does_not_stop_live_worker(monkeypatch, tmp_path, status):
    sandbox, record, commands, _ = monitor_fixture(
        monkeypatch, tmp_path, stale=False,
        relay_events=[{"path": "/chat/completions", "status": status}],
    )
    outcomes = iter([None, 0])
    sandbox.process.get_session_command = lambda *a, **k: SimpleNamespace(exit_code=next(outcomes))
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["controller_exit"] == 0
    assert "provider_transport_failure" not in record
    assert not any("docker kill" in command for command in commands)


def test_failed_transport_stop_is_not_swallowed_as_an_observation_error(monkeypatch, tmp_path):
    sandbox, record, _, _ = monitor_fixture(
        monkeypatch, tmp_path, stale=False,
        relay_events=[{"status": "transport_error", "error_type": "IncompleteRead"}],
    )
    sandbox.process.exec = lambda *a, **k: SimpleNamespace(exit_code=1, result="stop failed")
    with pytest.raises(RuntimeError, match="stop failed"):
        daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["provider_transport_failure"]["error_type"] == "IncompleteRead"
    assert not record.get("terminal_confirmed")


def test_stale_running_status_cannot_hide_a_dead_controller(monkeypatch, tmp_path):
    sandbox, record, commands, _ = monitor_fixture(monkeypatch, tmp_path, running=False, exit_code=137)
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["terminal_confirmed"] is True
    assert record["controller_exit"] == 137
    assert record["termination_reason"] == "controller_exited"
    assert any("docker kill" in command for command in commands)


def test_provider_admissions_refresh_activity_without_lowering_turn_limits(monkeypatch, tmp_path):
    sandbox, record, commands, clock = monitor_fixture(monkeypatch, tmp_path)
    sandbox.fs.download_file = lambda path, *a: (
        b'{"generation_requests":2,"input_counter_requests":0}'
        if path.endswith("ledger.json") else b'{"status":"running"}'
    )
    outcomes = iter([None, 0])
    sandbox.process.get_session_command = lambda *a, **k: SimpleNamespace(exit_code=next(outcomes))
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["activity_counters"] == [2, 0]
    assert record["last_activity_at"] == 10000
    assert record["controller_exit"] == 0
    assert not any("docker kill" in command for command in commands)


def test_created_controller_is_not_mistaken_for_a_dead_controller(monkeypatch, tmp_path):
    sandbox, record, commands, _ = monitor_fixture(monkeypatch, tmp_path, stale=False)
    original = sandbox.process.exec

    def execute(command, **kwargs):
        if "docker inspect" in command:
            return SimpleNamespace(exit_code=0, result='{"Running":false,"Status":"created","ExitCode":0}')
        return original(command, **kwargs)

    outcomes = iter([None, 0])
    sandbox.process.exec = execute
    sandbox.process.get_session_command = lambda *a, **k: SimpleNamespace(exit_code=next(outcomes))
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["controller_exit"] == 0
    assert "termination_reason" not in record
    assert not (tmp_path / "executor-terminal-proof.json").exists()


def test_quiet_observed_executor_keeps_supervisor_heartbeat_current(monkeypatch, tmp_path):
    sandbox, record, commands, clock = monitor_fixture(monkeypatch, tmp_path, stale=False)
    snapshots = []

    def observe(*args, **kwargs):
        return SimpleNamespace(exit_code=0 if clock[0] >= 10700 else None)

    sandbox.process.get_session_command = observe
    daytona._monitor_executor(
        sandbox, record, tmp_path,
        save=lambda: snapshots.append((clock[0], record.get("last_observed_at"))),
    )
    assert record["controller_exit"] == 0
    assert record["activity_counters"] == [1, 0]
    assert record["last_activity_at"] == 10000
    observed = [(at, heartbeat) for at, heartbeat in snapshots if heartbeat is not None]
    assert observed[-1][1] >= 10690
    assert all(at - heartbeat <= 10 for at, heartbeat in observed)
    assert not any("docker kill" in command for command in commands)


def test_guarded_executor_is_stopped_at_its_absolute_deadline(monkeypatch, tmp_path):
    sandbox, record, commands, _ = monitor_fixture(monkeypatch, tmp_path, stale=False)
    record["deadline_at"] = 9999
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["termination_reason"] == "executor_deadline_exceeded"
    assert any("docker kill" in command for command in commands)


def test_failed_stop_does_not_claim_terminal_ownership(monkeypatch, tmp_path):
    sandbox, record, _, _ = monitor_fixture(monkeypatch, tmp_path)
    sandbox.process.exec = lambda *a, **k: SimpleNamespace(exit_code=1, result="stop failed")
    with pytest.raises(RuntimeError, match="stop failed"):
        daytona._stop_stalled_executor(sandbox, record, tmp_path, reason="executor_stalled", save=lambda: None)
    assert not record.get("terminal_confirmed")


@pytest.mark.parametrize("value", [True, 0, 1799, 86401, float("nan")])
def test_invalid_stall_policy_is_rejected_before_allocating_compute(value):
    with pytest.raises(ValueError, match="stall seconds"):
        daytona._health_policy({"runtime": {"executor_stall_seconds": value}})


def test_guarded_missing_executor_is_confirmed_lost_without_starting_another(tmp_path):
    from daytona.common.errors import DaytonaNotFoundError
    record = {
        "sandbox_id": "original-id", "phase": "controller", "controller_started": True,
        "planned_conditions": 1, "health_policy": daytona._health_policy({}),
    }
    (tmp_path / "daytona-delivery.json").write_text(json.dumps(record))

    class Client:
        def get(self, identity):
            assert identity == "original-id"
            raise DaytonaNotFoundError("deleted")

    manifest = {"assets": [{"id": "box"}], "harnesses": [{"id": "pi"}],
                "models": [{"id": "selected", "model_id": "fixture-model"}],
                "skill_sets": [{"id": "none"}], "seed": 0, "repetitions": 1}
    result = daytona.reconnect_daytona(manifest, policy={"providers": {}}, output_dir=tmp_path, client=Client())
    delivery = result["deployment"]
    assert delivery["collection"] == "lost" and delivery["cleanup_confirmed"]
    assert delivery["missing_executor_proof"]["sandbox_id"] == "original-id"
    assert delivery["terminal_confirmed"] is True
    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "infrastructure_error"
    assert result["results"][0].get("simready_reward") is None


def test_public_cli_passes_the_frozen_executor_wheel(monkeypatch, tmp_path):
    from simready_bench import benchmark
    from simready_bench.cli import main
    captured = {}

    def execute(reference, **options):
        captured.update(options)
        return {"completed_conditions": 1, "planned_conditions": 1, "results": []}

    monkeypatch.setattr(benchmark, "benchmark", execute)
    wheel = tmp_path / "simready_bench-1.0.0-py3-none-any.whl"
    assert main(["benchmark", "NV25", "--model", "fixture", "--deploy", "daytona",
                 "--output-dir", str(tmp_path / "run"), "--wheel", str(wheel)]) == 0
    assert captured["wheel"] == wheel
