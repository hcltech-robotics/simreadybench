"""Exercise the release gate without running Docker or loading runtime images."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
IMAGES = {
    "actor": "sha256:" + "1" * 64,
    "verifier": "sha256:" + "2" * 64,
    "cpu": "sha256:" + "3" * 64,
}
OUTER = "simreadybench/runtime:consumer-check"


@pytest.fixture
def consumer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "containers/daytona"))
    spec = importlib.util.spec_from_file_location(
        "runtime_consumer_release", ROOT / "containers/daytona/release_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = SimpleNamespace(
        calls=[],
        ready=[0],
        running="true",
        actual={},
        missing=None,
        failed_image=None,
        launch_failure=False,
        logs_failure=False,
    )

    def invoke(command, **kwargs):
        command = [str(value) for value in command]
        state.calls.append(command)
        if command[:3] == ["docker", "manifest", "inspect"]:
            return subprocess.CompletedProcess(command, 1)
        if command[:2] == ["docker", "run"]:
            if state.launch_failure:
                raise subprocess.CalledProcessError(125, command)
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "logs"]:
            if state.logs_failure:
                raise OSError("Docker log collection failed")
            return subprocess.CompletedProcess(command, 0, b"consumer bootstrap log\n")
        if command[:2] == ["docker", "exec"] and command[3:] == [
            "docker", "info"
        ]:
            code = state.ready.pop(0) if len(state.ready) > 1 else state.ready[0]
            if isinstance(code, Exception):
                raise code
            return subprocess.CompletedProcess(command, code)
        if command[:2] == ["docker", "exec"] and command[3:5] == [
            "docker", "run"
        ]:
            if state.failed_image in command and state.failed_image is not None:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(f"Unexpected process: {command!r}")

    def capture(command, **kwargs):
        command = [str(value) for value in command]
        state.calls.append(command)
        if command[:2] == ["docker", "inspect"]:
            return state.running + "\n"
        if command[:2] == ["docker", "exec"] and command[3:6] == [
            "docker", "image", "inspect"
        ]:
            identifier = command[6]
            if identifier == state.missing:
                raise subprocess.CalledProcessError(1, command)
            return state.actual.get(identifier, identifier) + "\n"
        raise AssertionError(f"Unexpected capture: {command!r}")

    monkeypatch.setattr(module.subprocess, "run", invoke)
    monkeypatch.setattr(module.subprocess, "check_output", capture)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module, state


def assert_owned_cleanup(state):
    launch = next(call for call in state.calls if call[:2] == ["docker", "run"])
    name = launch[launch.index("--name") + 1]
    assert name.startswith("simready-runtime-check-")
    removals = [call for call in state.calls if call[:2] == ["docker", "rm"]]
    assert removals == [["docker", "rm", "--force", "--volumes", name]]
    assert not any("prune" in call for call in state.calls)
    return name


def test_consumer_waits_then_runs_all_exact_images_offline(consumer, tmp_path):
    module, state = consumer
    state.ready = [1, 0]
    receipt = module.verify_runtime(OUTER, IMAGES, tmp_path)

    assert receipt["status"] == "passed"
    assert receipt["images"] == IMAGES
    assert json.loads((tmp_path / "consumer-check.json").read_text()) == receipt
    assert (tmp_path / "consumer-check.log").read_bytes() == b"consumer bootstrap log\n"
    name = assert_owned_cleanup(state)
    launch = state.calls[0]
    assert launch[-1] == OUTER
    assert "--privileged" in launch
    for option, expected in (("--network", "none"), ("--cpus", "2"), ("--memory", "6g")):
        assert launch[launch.index(option) + 1] == expected
    assert [call for call in state.calls if call[3:5] == ["docker", "run"]] == [
        ["docker", "exec", name, "docker", "run", "--rm", "--network", "none", identifier, "/bin/true"]
        for identifier in IMAGES.values()
    ]
    assert len([call for call in state.calls if call[-2:] == ["docker", "info"]]) == 2


@pytest.mark.parametrize("role", IMAGES)
def test_consumer_rejects_actual_identity_mismatch(consumer, tmp_path, role):
    module, state = consumer
    state.actual[IMAGES[role]] = "sha256:" + "f" * 64
    with pytest.raises(RuntimeError, match=role + " image has another identity"):
        module.verify_runtime(OUTER, IMAGES, tmp_path)

    assert not (tmp_path / "consumer-check.json").exists()
    assert_owned_cleanup(state)
    assert not any(
        call[3:5] == ["docker", "run"] and IMAGES[role] in call
        for call in state.calls
    )


def test_consumer_missing_expected_image_cannot_pass(consumer, tmp_path):
    module, state = consumer
    state.missing = IMAGES["verifier"]
    with pytest.raises(subprocess.CalledProcessError):
        module.verify_runtime(OUTER, IMAGES, tmp_path)
    assert not (tmp_path / "consumer-check.json").exists()
    assert_owned_cleanup(state)


def test_consumer_failed_image_execution_cannot_pass(consumer, tmp_path):
    module, state = consumer
    state.failed_image = IMAGES["cpu"]
    with pytest.raises(subprocess.CalledProcessError):
        module.verify_runtime(OUTER, IMAGES, tmp_path)
    assert not (tmp_path / "consumer-check.json").exists()
    assert_owned_cleanup(state)


def test_consumer_exited_bootstrap_is_retained_and_removed(consumer, tmp_path):
    module, state = consumer
    state.ready = [1]
    state.running = "false"
    with pytest.raises(RuntimeError, match="did not finish loading"):
        module.verify_runtime(OUTER, IMAGES, tmp_path)
    assert not (tmp_path / "consumer-check.json").exists()
    assert (tmp_path / "consumer-check.log").is_file()
    assert_owned_cleanup(state)


def test_consumer_loading_deadline_still_cleans_up(consumer, tmp_path, monkeypatch):
    module, state = consumer
    state.ready = [1]
    ticks = iter((0, 601))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="did not finish loading"):
        module.verify_runtime(OUTER, IMAGES, tmp_path, timeout=600)
    assert not (tmp_path / "consumer-check.json").exists()
    assert_owned_cleanup(state)


def test_consumer_readiness_process_timeout_still_cleans_up(consumer, tmp_path):
    module, state = consumer
    state.ready = [subprocess.TimeoutExpired("docker info", 30)]
    with pytest.raises(subprocess.TimeoutExpired):
        module.verify_runtime(OUTER, IMAGES, tmp_path)
    assert not (tmp_path / "consumer-check.json").exists()
    assert_owned_cleanup(state)


def test_consumer_log_failure_cannot_skip_owned_cleanup(consumer, tmp_path):
    module, state = consumer
    state.ready = [1]
    state.running = "false"
    state.logs_failure = True
    with pytest.raises((RuntimeError, OSError)):
        module.verify_runtime(OUTER, IMAGES, tmp_path)
    assert_owned_cleanup(state)


def test_consumer_log_disk_failure_cannot_skip_owned_cleanup(
    consumer, tmp_path, monkeypatch
):
    module, state = consumer
    state.ready = [1]
    state.running = "false"
    original = Path.write_bytes

    def write_bytes(path, data):
        if path.name == "consumer-check.log":
            raise OSError(28, "No space left on device")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    with pytest.raises((RuntimeError, OSError)):
        module.verify_runtime(OUTER, IMAGES, tmp_path)
    assert_owned_cleanup(state)


def test_consumer_failed_launch_reconciles_its_unique_container(consumer, tmp_path):
    module, state = consumer
    state.launch_failure = True
    with pytest.raises(subprocess.CalledProcessError):
        module.verify_runtime(OUTER, IMAGES, tmp_path)
    assert not (tmp_path / "consumer-check.json").exists()
    assert_owned_cleanup(state)


def test_publish_does_not_push_when_consumer_loads_wrong_identity(
    consumer, tmp_path, monkeypatch
):
    module, state = consumer
    state.actual[IMAGES["actor"]] = "sha256:" + "f" * 64
    original_run = module.run

    def run(command):
        command = [str(value) for value in command]
        if command[:2] in (["docker", "run"], ["docker", "exec"]):
            return original_run(command)
        state.calls.append(command)

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(module, "prepare", lambda **kwargs: IMAGES)
    wheel = tmp_path / "simready_bench-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel fixture")
    output = tmp_path / "release"

    with pytest.raises(RuntimeError, match="actor image has another identity"):
        module.publish(
            repository="ghcr.io/example/simreadybench-runtime",
            version="1.0.1",
            wheel=wheel,
            output=output,
            revision="a" * 40,
        )

    assert not any(call[:2] == ["docker", "push"] for call in state.calls)
    assert not (output / "runtime-release.json").exists()
    assert not (output / "consumer-check.json").exists()
    assert_owned_cleanup(state)
