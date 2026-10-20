"""Complete submitted actors after their final snapshot has been captured."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError, ApiRateLimitError
from harbor.agents.installed.pi import Pi
from harbor.environments.docker.docker import DockerEnvironment
from simready_bench.local import bounded_controller_class
from simready_bench.engine.harbor import NativeLifecycle, BenchmarkDockerEnvironment
from simready_bench.engine.pi import PiAgent


def setup(tmp_path, invalid=False):
    actor = tmp_path / "actor"
    (actor / "output").mkdir(parents=True)
    if not invalid:
        (actor / "output/asset.usd").write_text(
            '#usda 1.0\n(defaultPrim="Box")\ndef Xform "Box" { def Cube "Visual" { double size=1 } }\n'
        )
    else:
        (actor / "output/alternate.usda").write_text('#usda 1.0\ndef Xform "Part" {}\n')
    controller = bounded_controller_class()(tmp_path / "controller")
    controller.register_episode(
        "episode",
        actor,
        task_id="task",
        trial_id="episode",
        runner="harbor",
        deadline_seconds=60,
        inference_limit=3,
        validation_limit=1,
        preview_limit=1,
    )
    controller.start_episode("episode")
    bridge = NativeLifecycle(controller, tmp_path / "actors", stop_timeout_seconds=1)
    bridge.install()
    env = object.__new__(BenchmarkDockerEnvironment)
    env.bridge = bridge
    env.episode_id = "episode"
    env.actor_root = actor
    env.actor_stopped = False
    env.actor_phase = True
    env.actor_start_attempted = True
    env.final_snapshot = None
    env.network_isolation_proof = {"container_id": "a" * 64}
    env._actor_stop_audit = None
    bridge._episodes["episode"] = env
    agent = object.__new__(PiAgent)
    agent.native_bridge_id = bridge.bridge_id
    agent._native_bound_episode = "episode"
    bridge._deferred_agents.append(agent)
    return controller, bridge, env, agent


@pytest.mark.parametrize("invalid", [False, True])
def test_submit_waits_for_real_durable_snapshot_and_only_then_returns_normally(
    tmp_path, monkeypatch, invalid
):
    async def exercise():
        controller, bridge, env, agent = setup(tmp_path, invalid)
        killed = asyncio.Event()
        release = asyncio.Event()

        async def stop():
            env.actor_stopped = True
            env._actor_stop_audit = {
                "container_id": "a" * 64,
                "kill_return_code": 0,
                "was_running": True,
                "state": {
                    "Running": False,
                    "Paused": False,
                    "OOMKilled": False,
                    "ExitCode": 137,
                },
            }
            killed.set()
            await release.wait()

        async def docker_exec(*args, **kwargs):
            await killed.wait()
            return SimpleNamespace(return_code=137, stdout="", stderr="")

        async def native_run(self, instruction, environment, context):
            await environment.exec("pi --print --mode json")

        monkeypatch.setattr(env, "stop_actor_writes", stop)
        monkeypatch.setattr(DockerEnvironment, "exec", docker_exec)
        monkeypatch.setattr(Pi, "run", native_run)
        before = controller.get_episode("episode")
        controller.close_episode("episode", "submitted")
        stopping = asyncio.create_task(bridge.request_stop("episode", "submitted"))
        running = asyncio.create_task(
            agent.run("unchanged instruction", env, SimpleNamespace())
        )
        await killed.wait()
        await asyncio.sleep(0.03)
        try:
            assert not running.done(), (
                "Pi must await the existing submitted stop, not fail on its intentional137"
            )
            release.set()
            assert await stopping is True
            await running
            final = controller.get_episode("episode")
            assert final.termination_reason == "submitted"
            with controller._connect() as db:
                final_snapshot_id = db.execute(
                    "SELECT final_snapshot_id FROM episodes WHERE episode_id=?",
                    ("episode",),
                ).fetchone()[0]
            assert final_snapshot_id == env.final_snapshot.snapshot_id
            assert env.final_snapshot.valid is (not invalid)
            if not invalid:
                assert (Path(env.final_snapshot.path) / "asset.usd").read_bytes() == (
                    env.actor_root / "output/asset.usd"
                ).read_bytes()
            assert final.inference_calls_used == before.inference_calls_used == 0
            assert final.validation_calls_used == before.validation_calls_used == 0
            assert final.deadline_at == before.deadline_at
            assert agent.actor_stop_receipt["raw_exit_code"] == 137
            assert agent.actor_stop_receipt["snapshot_id"] == final_snapshot_id
        finally:
            release.set()
            await asyncio.gather(stopping, running, return_exceptions=True)
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "error",
    [
        NonZeroAgentExitCodeError("raw137"),
        ApiRateLimitError("rate limit"),
        RuntimeError("freeze failed"),
    ],
)
def test_untyped_or_classified_failures_are_never_swallowed(
    tmp_path, monkeypatch, error
):
    async def exercise():
        _, bridge, env, agent = setup(tmp_path)

        async def native_run(*args, **kwargs):
            raise error

        monkeypatch.setattr(Pi, "run", native_run)
        try:
            with pytest.raises(type(error), match=str(error)):
                await agent.run("", env, SimpleNamespace())
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "reason",
    ["cancelled", "interrupted_budget", "agent_completed", "infrastructure_disk_guard"],
)
def test_unrelated_stops_never_become_controlled_completion(
    tmp_path, monkeypatch, reason
):
    async def exercise():
        c, b, e, a = setup(tmp_path)

        async def stop():
            e.actor_stopped = True
            e._actor_stop_audit = {
                "container_id": "a" * 64,
                "kill_return_code": 0,
                "was_running": True,
                "state": {
                    "Running": False,
                    "Paused": False,
                    "OOMKilled": False,
                    "ExitCode": 137,
                },
            }

        monkeypatch.setattr(e, "stop_actor_writes", stop)
        try:
            await b.request_stop("episode", reason)
            assert await b.confirmed_actor_exit(e, 137) is None
        finally:
            b.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "change",
    [
        "oom",
        "kill_failed",
        "was_stopped",
        "wrong_container",
        "wrong_episode",
        "no_freeze",
    ],
)
def test_stop_evidence_failures_cannot_authorize_success(tmp_path, monkeypatch, change):
    async def exercise():
        c, b, e, a = setup(tmp_path)

        async def stop():
            e.actor_stopped = True
            audit = {
                "container_id": "a" * 64,
                "kill_return_code": 0,
                "was_running": True,
                "state": {
                    "Running": False,
                    "Paused": False,
                    "OOMKilled": False,
                    "ExitCode": 137,
                },
            }
            if change == "oom":
                audit["state"]["OOMKilled"] = True
            if change == "kill_failed":
                audit["kill_return_code"] = 1
            if change == "was_stopped":
                audit["was_running"] = False
            if change == "wrong_container":
                audit["container_id"] = "b" * 64
            e._actor_stop_audit = audit

        monkeypatch.setattr(e, "stop_actor_writes", stop)
        if change == "no_freeze":

            def broken(*args, **kwargs):
                raise RuntimeError("snapshot disk error")

            monkeypatch.setattr(c, "finalize_submission", broken)
        c.close_episode("episode", "submitted")
        try:
            if change == "no_freeze":
                with pytest.raises(RuntimeError, match="snapshot disk error"):
                    await b.request_stop("episode", "submitted")
            else:
                await b.request_stop("episode", "submitted")
            if change == "wrong_episode":
                e.episode_id = "other"
            assert await b.confirmed_actor_exit(e, 137) is None
            assert not getattr(a, "actor_stop_receipt", None)
        finally:
            b.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "corrupt", ["snapshot", "db_record", "agent", "bridge", "raw_code"]
)
def test_exact_receipt_binding_and_raw_exit_cannot_be_reused_for_other_state(
    tmp_path, monkeypatch, corrupt
):
    async def exercise():
        from dataclasses import replace

        c, b, e, a = setup(tmp_path)

        async def stop():
            e.actor_stopped = True
            e._actor_stop_audit = {
                "container_id": "a" * 64,
                "kill_return_code": 0,
                "was_running": True,
                "state": {
                    "Running": False,
                    "Paused": False,
                    "OOMKilled": False,
                    "ExitCode": 137,
                },
            }

        monkeypatch.setattr(e, "stop_actor_writes", stop)
        c.close_episode("episode", "submitted")
        try:
            await b.request_stop("episode", "submitted")
            receipt = await b.confirmed_actor_exit(e, 137)
            assert receipt is not None
            from simready_bench.engine.harbor import NativeActorStop

            signal = NativeActorStop(receipt)
            assert b.accept_actor_stop(a, e, signal)
            if corrupt == "snapshot":
                e.final_snapshot = replace(e.final_snapshot, sha256="0" * 64)
                assert await b.confirmed_actor_exit(e, 137) is None
            elif corrupt == "db_record":
                with c._connect() as db:
                    db.execute(
                        "UPDATE episodes SET final_snapshot_id=NULL WHERE episode_id=?",
                        ("episode",),
                    )
                assert await b.confirmed_actor_exit(e, 137) is None
            elif corrupt == "agent":
                a._native_bound_episode = "other"
                assert not b.accept_actor_stop(a, e, signal)
            elif corrupt == "bridge":
                a.native_bridge_id = "other"
                assert not b.accept_actor_stop(a, e, signal)
            else:
                assert await b.confirmed_actor_exit(e, 1) is None
            forged = NativeActorStop(replace(receipt))
            assert not b.accept_actor_stop(a, e, forged)
        finally:
            b.uninstall()

    asyncio.run(exercise())


def test_pending_wait_timeout_does_not_cancel_or_start_stop(tmp_path):
    async def exercise():
        c, b, e, a = setup(tmp_path)
        b.stop_timeout_seconds = 0.001
        c.snapshot_timeout_seconds = 0.001
        pending = asyncio.get_running_loop().create_future()
        b._controlled_stops["episode"] = (e, pending)
        try:
            with pytest.raises(TimeoutError):
                await b.confirmed_actor_exit(e, 137)
            assert not pending.cancelled()
            assert not e.actor_stopped and e.final_snapshot is None
            assert c.get_episode("episode").status == "active"
        finally:
            pending.set_result(None)
            b.uninstall()

    asyncio.run(exercise())


def test_actual_stop_method_preserves_engine_evidence(tmp_path, monkeypatch):
    async def exercise():
        c, b, e, a = setup(tmp_path)
        states = iter(
            [
                {"Running": True, "Paused": False, "OOMKilled": False, "ExitCode": 0},
                {
                    "Running": False,
                    "Paused": False,
                    "OOMKilled": False,
                    "ExitCode": 137,
                },
            ]
        )
        commands = []

        async def compose(args, **kwargs):
            commands.append(list(args))
            return SimpleNamespace(
                return_code=0, stdout="a" * 64 if args[0] == "ps" else ""
            )

        async def engine(*args):
            import json

            assert args == ("inspect", "--format", "{{json .State}}", "a" * 64)
            return json.dumps(next(states)).encode()

        monkeypatch.setattr(e, "_run_docker_compose_command", compose)
        monkeypatch.setattr(e, "_engine", engine)
        try:
            await e.stop_actor_writes()
            assert e.actor_stopped is True
            assert commands == [
                ["ps", "--all", "--quiet", "main"],
                ["kill", "-s", "SIGKILL", "main"],
            ]
            assert e._actor_stop_audit["kill_return_code"] == 0
            assert e._actor_stop_audit["was_running"] is True
            assert e._actor_stop_audit["state"]["ExitCode"] == 137
        finally:
            b.uninstall()

    asyncio.run(exercise())


def assert_stock_pi_delegation(cls):
    """The thin adapter must await stock Pi.run once with identical objects."""
    from unittest.mock import patch
    from harbor.agents.installed.pi import Pi

    calls = []
    instruction = object()
    environment = object()
    context = object()
    sentinel = object()

    async def stock(self, *args):
        calls.append((self, args))
        return sentinel

    agent = object.__new__(cls)
    with patch.object(Pi, "run", stock):
        assert asyncio.run(agent.run(instruction, environment, context)) is sentinel
    assert calls == [(agent, (instruction, environment, context))]


@pytest.mark.parametrize("subclass_signal", [False, True])
def test_wrapper_never_accepts_forged_or_subclassed_signal(
    tmp_path, monkeypatch, subclass_signal
):
    async def exercise():
        from simready_bench.engine.harbor import NativeActorStop

        c, b, e, a = setup(tmp_path)

        class ForgedSubclass(NativeActorStop):
            pass

        signal = (ForgedSubclass if subclass_signal else NativeActorStop)(object())

        async def native(*args):
            raise signal

        monkeypatch.setattr(Pi, "run", native)
        try:
            with pytest.raises(type(signal)):
                await a.run("", e, SimpleNamespace())
        finally:
            b.uninstall()

    asyncio.run(exercise())


def test_preinspection_failure_never_skips_mandatory_actor_kill(tmp_path, monkeypatch):
    async def exercise():
        import json

        c, b, e, a = setup(tmp_path)
        calls = []
        inspections = 0

        async def compose(args, **kwargs):
            calls.append(list(args))
            return SimpleNamespace(
                return_code=0, stdout="a" * 64 if args[0] == "ps" else ""
            )

        async def engine(*args):
            nonlocal inspections
            inspections += 1
            if inspections == 1:
                raise RuntimeError("preinspection unavailable")
            return json.dumps(
                {"Running": False, "Paused": False, "OOMKilled": False, "ExitCode": 137}
            ).encode()

        monkeypatch.setattr(e, "_run_docker_compose_command", compose)
        monkeypatch.setattr(e, "_engine", engine)
        try:
            c.close_episode("episode", "submitted")
            await b.request_stop("episode", "submitted")
            assert calls == [
                ["ps", "--all", "--quiet", "main"],
                ["kill", "-s", "SIGKILL", "main"],
            ]
            assert e.actor_stopped is True
            assert e._actor_stop_audit["was_running"] is False
            assert await b.confirmed_actor_exit(e, 137) is None
        finally:
            b.uninstall()

    asyncio.run(exercise())
