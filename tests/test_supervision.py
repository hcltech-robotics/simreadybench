import errno
import json
import os
from pathlib import Path
import shutil
import signal
import ssl
import subprocess
import sys
import time
import urllib.error

import pytest

from simready_bench import cli, supervision
from simready_bench.daytona import DaytonaObservationError
from simready_bench.publishing import PublicationError
from simready_bench.recovery import RecoveryLimitReached


def test_child_arguments_preserve_benchmark_options_and_remove_supervision():
    arguments = [
        "benchmark", "NV25", "--model", "gpt-6.1-sol", "--authoring-turns", "64",
        "--supervise", "--coordinator-restarts=5", "--workers", "4",
    ]
    assert supervision._child_arguments(arguments) == [
        "benchmark", "NV25", "--model", "gpt-6.1-sol", "--authoring-turns", "64",
        "--workers", "4",
    ]
    assert supervision._child_arguments(["benchmark", "--coordinator-restarts", "5"]) == ["benchmark"]


def test_invocation_identity_excludes_only_resume_and_worker_count():
    original = ["benchmark", "NV25", "--model", "fixture", "--authoring-turns", "64", "--policy", "operator.json"]
    assert supervision._invocation_arguments([*original, "--workers", "4"]) == original
    assert supervision._invocation_arguments([*original, "--workers=2", "--resume"]) == original


@pytest.mark.parametrize(
    "error,expected",
    [
        (DaytonaObservationError("temporary observation failure"), 75),
        (ConnectionResetError("connection reset"), 75),
        (TimeoutError("timeout"), 75),
        (PublicationError("network publication failure", retryable=True), 75),
        (PublicationError("changed completed row", retryable=False), 2),
        (RecoveryLimitReached("reconstruction retries exhausted"), 2),
        (OSError(errno.ENOSPC, "disk full"), 2),
        (ssl.SSLCertVerificationError(1, "certificate failed"), 2),
        (urllib.error.HTTPError("https://fixture.invalid", 401, "unauthorised", {}, None), 2),
        (ValueError("changed frozen manifest"), 2),
    ],
)
def test_only_retryable_daytona_failures_use_temporary_exit(error, expected):
    assert supervision.coordinator_exit_code(error, deploy="daytona") == expected
    assert supervision.coordinator_exit_code(error, deploy="local") == 2


def test_cli_supervision_has_a_pure_dispatch_boundary(tmp_path, monkeypatch):
    captured = []
    arguments = [
        "benchmark", "NV25", "--model", "gpt-6.1-sol", "--deploy", "daytona",
        "--output-dir", str(tmp_path / "new-run"), "--authoring-turns", "64",
        "--supervise", "--coordinator-restarts", "5",
    ]
    monkeypatch.setattr(supervision, "supervise", lambda argv, **options: captured.append((argv, options)) or 0)
    assert cli.main(arguments) == 0
    assert captured == [(arguments, {"output_dir": tmp_path / "new-run", "maximum_restarts": 5})]
    assert not (tmp_path / "new-run").exists()


def test_supervision_requires_daytona(tmp_path):
    with pytest.raises(SystemExit) as caught:
        cli.main(["benchmark", "--model", "gpt-6.1-sol", "--output-dir", str(tmp_path), "--supervise"])
    assert caught.value.code == 2


def test_unsupported_platform_fails_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision.sys, "platform", "win32")
    with pytest.raises(ValueError, match="Linux or WSL"):
        supervision.supervise(["benchmark"], output_dir=tmp_path / "new-run")
    assert not (tmp_path / "new-run").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")
def test_manager_lock_does_not_precreate_benchmark_output(tmp_path, monkeypatch):
    output = tmp_path / "fresh-output"

    class Child:
        pid = 12345

        def __init__(self, *args, **options):
            assert not output.exists()

        def wait(self):
            return 2

    monkeypatch.setattr(supervision.subprocess, "Popen", Child)
    assert supervision.supervise(["benchmark"], output_dir=output) == 2
    assert not output.exists()
    assert (tmp_path / ".fresh-output.supervisor.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")
@pytest.mark.parametrize("code", [75, -getattr(signal, "SIGKILL", 9)])
def test_failure_without_frozen_manifest_cannot_launch_another_child(
    tmp_path, monkeypatch, code
):
    output = tmp_path / "uninitialised-output"
    launches = []

    class Child:
        pid = 12345

        def __init__(self, *args, **options):
            launches.append(args)

        def wait(self):
            return code

    monkeypatch.setattr(supervision.subprocess, "Popen", Child)
    assert supervision.supervise(["benchmark"], output_dir=output) == 75
    assert len(launches) == 1 and not output.exists()
    state = json.loads((tmp_path / ".uninitialised-output.supervisor.json").read_text())
    assert state["status"] == "not_resumable" and state["restarts"] == 0


def saved_benchmark(output, *, turns=64):
    output.mkdir(exist_ok=True)
    experiment, operator = {"turns": turns}, {"route": "fixture"}
    for name, value in (("experiment.json", experiment), ("operator.json", operator)):
        (output / name).write_text(json.dumps(value), encoding="ascii")
    (output / "run.json").write_text(
        json.dumps({"schema": "simready-bench-run-v1", "deployment": "daytona", "experiment_sha256": supervision._digest(experiment), "operator_sha256": supervision._digest(operator)}),
        encoding="ascii",
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision")
def test_consistent_but_changed_saved_manifest_cannot_be_restarted(tmp_path, monkeypatch):
    output = tmp_path / "saved-output"
    saved_benchmark(output)
    launches, clock = [], [100.0]

    class Child:
        pid = 12345

        def __init__(self, *args, **options):
            launches.append(args)

        def wait(self):
            return 75

    def sleep(seconds):
        clock[0] += seconds
        saved_benchmark(output, turns=128)

    monkeypatch.setattr(supervision.subprocess, "Popen", Child)
    monkeypatch.setattr(supervision.time, "time", lambda: clock[0])
    monkeypatch.setattr(supervision.time, "sleep", sleep)
    with pytest.raises(ValueError, match="saved benchmark identity changed"):
        supervision.supervise(["benchmark"], output_dir=output)
    assert len(launches) == 1


def fixture_command(tmp_path, *, mode, maximum_restarts=1):
    package = tmp_path / "fixture-package/simready_bench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="ascii")
    shutil.copyfile(supervision.__file__, package / "supervision.py")
    (package / "__main__.py").write_text(
        '''import hashlib,json,os,signal,sys,time
from pathlib import Path
args=sys.argv[1:]
output=Path(args[args.index('--output-dir')+1])
output.mkdir(exist_ok=True)
def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
experiment={'model':'fixture','turns':64}
operator={'route':'fixture'}
for name,value in [('experiment.json',experiment),('operator.json',operator)]:
 (output/name).write_text(json.dumps(value))
(output/'run.json').write_text(json.dumps({'schema':'simready-bench-run-v1','deployment':'daytona','experiment_sha256':digest(experiment),'operator_sha256':digest(operator)}))
record={'argv':args,'pid':os.getpid(),'ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}
with (output.parent/'children.jsonl').open('a') as stream:
 stream.write(json.dumps(record)+'\\n')
mode=os.environ['FIXTURE_MODE']
if mode=='incomplete': raise SystemExit(2)
if mode=='always75': raise SystemExit(75)
if '--resume' not in args:
 if mode=='signal': os.kill(os.getpid(),signal.SIGKILL)
 if mode=='hold': time.sleep(300)
 raise SystemExit(75)
raise SystemExit(0)
''',
        encoding="ascii",
    )
    runner = tmp_path / "run_supervisor.py"
    runner.write_text(
        '''import os,sys
from pathlib import Path
from simready_bench.supervision import SupervisorBusyError,supervise
try:
 code=supervise(sys.argv[1:],output_dir=Path(os.environ['FIXTURE_OUTPUT']),maximum_restarts=int(os.environ['FIXTURE_RESTARTS']))
except SupervisorBusyError:
 code=75
raise SystemExit(code)
''',
        encoding="ascii",
    )
    output = tmp_path / "benchmark-output"
    arguments = [
        "benchmark", "NV25", "--model", "fixture", "--deploy", "daytona",
        "--output-dir", str(output), "--authoring-turns", "64", "--workers", "4",
        "--supervise", "--coordinator-restarts", str(maximum_restarts),
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": str(package.parent),
        "FIXTURE_MODE": mode,
        "FIXTURE_OUTPUT": str(output),
        "FIXTURE_RESTARTS": str(maximum_restarts),
        "FIXTURE_API_KEY": "private-key-must-never-be-persisted",
    }
    return [sys.executable, str(runner), *arguments], environment, output


def records(output):
    return [json.loads(line) for line in (output.parent / "children.jsonl").read_text().splitlines()]


def wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("The local supervision fixture did not reach its expected state")


def process_matches(record):
    path = Path("/proc", str(record["pid"]), "stat")
    try:
        values = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return values[0] != "Z" and values[19] == record["ticks"]


@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux coordinator subprocesses")
@pytest.mark.parametrize("mode", ["retry", "signal"])
def test_actual_coordinator_failure_resumes_exact_saved_manifest(tmp_path, mode):
    command, environment, output = fixture_command(tmp_path, mode=mode)
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    children = records(output)
    assert len(children) == 2
    assert "--resume" not in children[0]["argv"] and "--resume" in children[1]["argv"]
    assert children[1]["argv"] == [*children[0]["argv"], "--resume"]
    state = json.loads((tmp_path / ".benchmark-output.supervisor.json").read_text())
    assert state["restarts"] == 1 and state["launches"] == 2
    assert state["status"] == "completed" and state["saved_run_sha256"]
    assert "private-key-must-never-be-persisted" not in json.dumps(state) + result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux coordinator subprocesses")
def test_relative_parent_output_supports_saved_coordinator_restart(tmp_path):
    command, environment, output = fixture_command(tmp_path, mode="retry")
    working = tmp_path / "nested"
    working.mkdir()
    relative = "../benchmark-output"
    command[command.index("--output-dir") + 1] = relative
    environment["FIXTURE_OUTPUT"] = relative
    result = subprocess.run(command, env=environment, cwd=working, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    children = records(output)
    assert len(children) == 2
    assert children[0]["argv"][children[0]["argv"].index("--output-dir") + 1] == relative
    assert children[1]["argv"] == [*children[0]["argv"], "--resume"]
    state = json.loads((tmp_path / ".benchmark-output.supervisor.json").read_text())
    assert state["status"] == "completed" and state["restarts"] == 1


@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux coordinator subprocesses")
def test_actual_clean_incomplete_is_terminal(tmp_path):
    command, environment, output = fixture_command(tmp_path, mode="incomplete")
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert len(records(output)) == 1
    state = json.loads((tmp_path / ".benchmark-output.supervisor.json").read_text())
    assert state["status"] == "incomplete" and state["restarts"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux coordinator subprocesses")
def test_actual_restart_limit_survives_manager_restart(tmp_path):
    command, environment, output = fixture_command(tmp_path, mode="always75")
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 75
    assert len(records(output)) == 2
    again = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    assert again.returncode == 75 and len(records(output)) == 2
    state = json.loads((tmp_path / ".benchmark-output.supervisor.json").read_text())
    assert state["status"] == "restart_exhausted" and state["restarts"] == 1


@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux parent-death signals and locks")
def test_one_manager_and_parent_death_prevent_orphan_coordinator(tmp_path):
    command, environment, output = fixture_command(tmp_path, mode="hold")
    manager = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = None
    try:
        wait_until(lambda: (tmp_path / "children.jsonl").exists())
        child = records(output)[0]
        assert process_matches(child)
        competing = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
        assert competing.returncode == 75 and len(records(output)) == 1
        manager.kill()
        manager.wait(timeout=5)
        wait_until(lambda: not process_matches(child))
        command[command.index("--workers") + 1] = "2"
        command.append("--resume")
        recovered = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
        assert recovered.returncode == 0, recovered.stderr
        assert len(records(output)) == 2 and "--resume" in records(output)[1]["argv"]
        resumed = records(output)[1]["argv"]
        assert resumed[resumed.index("--workers") + 1] == "2"
        state = json.loads((tmp_path / ".benchmark-output.supervisor.json").read_text())
        assert state["restarts"] == 1 and state["status"] == "completed"
    finally:
        if manager.poll() is None:
            manager.kill()
            manager.wait(timeout=5)
        if child and process_matches(child):
            os.kill(child["pid"], signal.SIGKILL)
        manager.communicate(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux coordinator subprocesses")
@pytest.mark.parametrize("option,value", [("--model", "other-model"), ("--authoring-turns", "32"), ("--policy", "changed-policy.json")])
def test_manager_resume_rejects_changed_scientific_arguments(tmp_path, option, value):
    command, environment, output = fixture_command(tmp_path, mode="incomplete")
    command.extend(["--policy", "original-policy.json"])
    first = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    assert first.returncode == 2 and len(records(output)) == 1
    command[command.index(option) + 1] = value
    command.append("--resume")
    again = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    assert again.returncode != 0
    assert "different benchmark command" in again.stderr
    assert len(records(output)) == 1
