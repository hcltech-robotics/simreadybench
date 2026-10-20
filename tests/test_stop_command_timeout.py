"""Preserve final submissions when Docker confirms a delayed stop."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from test_confirmed_submit_stop import setup


def test_default_stop_budget_allows_slow_docker_inspection(tmp_path, monkeypatch):
    from simready_bench.engine.harbor import NativeLifecycle

    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        default = NativeLifecycle(controller, tmp_path / "other-actors")
        assert default.stop_timeout_seconds == 120
        bridge.stop_timeout_seconds = default.stop_timeout_seconds
        docker_boundary(env, monkeypatch)
        original = env._engine
        calls = 0

        async def delayed(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(5.05)
            return await original(*args)

        monkeypatch.setattr(env, "_engine", delayed)
        try:
            assert await bridge.request_stop("episode", "submitted") is True
            assert bridge.completed_actor_stop(env)
            assert env.final_snapshot.valid
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


def final_id(controller):
    with controller._connect() as db:
        return db.execute(
            "SELECT final_snapshot_id FROM episodes WHERE episode_id='episode'"
        ).fetchone()[0]


def docker_boundary(env, monkeypatch, *, after=None, error=None, missing_before=False):
    state = (
        {
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "ExitCode": 137,
        }
        if after is None
        else after
    )
    inspections = 0

    async def compose(args, **kwargs):
        if args == ["ps", "--all", "--quiet", "main"]:
            return SimpleNamespace(return_code=0, stdout="a" * 64, stderr="")
        assert args == ["kill", "-s", "SIGKILL", "main"]
        assert kwargs == {"check": False, "timeout_sec": 30}
        # Docker has performed SIGKILL, but Harbor's command collector throws.
        raise (
            error
            if error is not None
            else RuntimeError("Command timed out after 30 seconds")
        )

    async def engine(*args):
        nonlocal inspections
        assert args == ("inspect", "--format", "{{json .State}}", "a" * 64)
        inspections += 1
        if inspections == 1:
            if missing_before:
                raise RuntimeError("preinspection unavailable")
            return json.dumps(
                {
                    "Running": True,
                    "Paused": False,
                    "Restarting": False,
                    "OOMKilled": False,
                    "ExitCode": 0,
                }
            ).encode()
        return json.dumps(state).encode()

    monkeypatch.setattr(env, "_run_docker_compose_command", compose)
    monkeypatch.setattr(env, "_engine", engine)


@pytest.mark.parametrize("reason", ["walltime_exhausted", "submitted"])
def test_kill_command_timeout_still_freezes_only_after_confirmed_owned_stop(
    tmp_path, monkeypatch, reason
):
    # Would fail if the timeout bypasses final state inspection and durable freeze.
    async def exercise():
        c, b, e, a = setup(tmp_path)
        before = c.get_episode("episode")
        docker_boundary(e, monkeypatch)
        try:
            assert await b.request_stop("episode", reason) is True
            assert e.actor_stopped is True and e.actor_phase is False
            assert e.final_snapshot.snapshot_id == final_id(c)
            assert e.final_snapshot.valid is True
            assert (e.actor_root / "output/asset.usd").read_bytes() == (
                __import__("pathlib").Path(e.final_snapshot.path) / "asset.usd"
            ).read_bytes()
            stopped = c.get_episode("episode")
            assert stopped.termination_reason == reason
            assert stopped.deadline_at == before.deadline_at
            assert stopped.inference_calls_used == before.inference_calls_used == 0
            assert stopped.validation_calls_used == before.validation_calls_used == 0
            assert stopped.preview_calls_used == before.preview_calls_used == 0
            assert e._actor_stop_audit["kill_return_code"] is None
            assert e._actor_stop_audit["kill_command_error"] == {
                "error_type": "RuntimeError",
                "code": "compose_kill_timeout",
            }
            receipt = await b.confirmed_actor_exit(e, 137)
            assert receipt is not None and b.completed_actor_stop(e)
            assert receipt.snapshot_id == final_id(c)
            assert list(
                (c.state_dir / "native-environments").glob("*.actor-stop-audit.json")
            )
            snapshot_id = e.final_snapshot.snapshot_id
            assert await b.request_stop("episode", reason) is True
            assert e.final_snapshot.snapshot_id == snapshot_id
        finally:
            b.uninstall()

    asyncio.run(exercise())


def test_submitted_pi_trial_retains_its_score_after_confirmed_kill_timeout(
    tmp_path, monkeypatch
):
    from harbor.agents.installed.pi import Pi
    from harbor.environments.docker.docker import DockerEnvironment
    from simready_bench.local import complete_trial_result
    from test_budget_completion import evaluate_final, native_result

    async def exercise():
        c, bridge, env, agent = setup(tmp_path)
        docker_boundary(env, monkeypatch)

        async def execute(*args, **kwargs):
            await bridge.request_stop("episode", "submitted")
            return SimpleNamespace(return_code=137, stdout="", stderr="")

        async def native_run(self, instruction, environment, context):
            await environment.exec("pi --print --mode json")

        monkeypatch.setattr(DockerEnvironment, "exec", execute)
        monkeypatch.setattr(Pi, "run", native_run)
        try:
            await agent.run("Reconstruct the asset.", env, SimpleNamespace())
            evaluation = evaluate_final(tmp_path, env.final_snapshot)
            result = complete_trial_result(
                {
                    "evaluation": evaluation,
                    "gateway_records": [
                        {"status": "completed", "upstream_status": 200}
                    ],
                },
                native_result("NonZeroAgentExitCodeError"),
                normal_stop=bridge.completed_actor_stop(env),
            )
            assert result["status"] == "completed"
            assert result["evaluation"] == evaluation
            assert agent.actor_stop_receipt["snapshot_id"] == final_id(c)
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "after",
    [
        {"Running": True, "Paused": False, "Restarting": False},
        {"Running": False, "Paused": True, "Restarting": False},
        {"Running": False, "Paused": False, "Restarting": True},
        {"Running": False, "Paused": False},
        {},
    ],
)
def test_timeout_never_freezes_unconfirmed_actor_state(tmp_path, monkeypatch, after):
    # Would fail if the recovery treats missing/paused/restarting state as stopped.
    async def exercise():
        c, b, e, a = setup(tmp_path)
        docker_boundary(e, monkeypatch, after=after)
        try:
            with pytest.raises(RuntimeError):
                await b.request_stop("episode", "walltime_exhausted")
            assert (
                not e.actor_stopped and e.final_snapshot is None and final_id(c) is None
            )
        finally:
            b.uninstall()

    asyncio.run(exercise())


def test_timeout_recovery_requires_original_bound_container_identity(
    tmp_path, monkeypatch
):
    async def exercise():
        c, b, e, a = setup(tmp_path)
        e.network_isolation_proof = {"container_id": "b" * 64}
        docker_boundary(e, monkeypatch)
        try:
            with pytest.raises(RuntimeError):
                await b.request_stop("episode", "walltime_exhausted")
            assert (
                not e.actor_stopped and e.final_snapshot is None and final_id(c) is None
            )
        finally:
            b.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "error", [RuntimeError("Docker permission denied"), asyncio.CancelledError()]
)
def test_other_errors_and_cancellation_are_not_timeout_recovery(
    tmp_path, monkeypatch, error
):
    async def exercise():
        c, b, e, a = setup(tmp_path)
        docker_boundary(e, monkeypatch, error=error)
        try:
            with pytest.raises(type(error)):
                await b.request_stop("episode", "walltime_exhausted")
            assert (
                not e.actor_stopped and e.final_snapshot is None and final_id(c) is None
            )
        finally:
            b.uninstall()

    asyncio.run(exercise())


def test_timeout_recovery_never_extends_stop_deadline(tmp_path, monkeypatch):
    async def exercise():
        c, b, e, a = setup(tmp_path)
        b.stop_timeout_seconds = 0.01
        docker_boundary(e, monkeypatch)
        original = e._engine
        count = 0

        async def engine(*args):
            nonlocal count
            count += 1
            if count == 2:
                await asyncio.sleep(0.05)
            return await original(*args)

        monkeypatch.setattr(e, "_engine", engine)
        try:
            with pytest.raises(TimeoutError):
                await b.request_stop("episode", "walltime_exhausted")
            assert (
                not e.actor_stopped and e.final_snapshot is None and final_id(c) is None
            )
        finally:
            b.uninstall()

    asyncio.run(exercise())
