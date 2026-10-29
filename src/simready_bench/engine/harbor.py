"Docker actor lifecycle and separate Harbor verification."

from __future__ import annotations
import asyncio
from contextlib import suppress, closing
from dataclasses import dataclass, asdict
from typing import ClassVar
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4, UUID
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments.docker.docker import DockerEnvironment
from harbor.trial.hooks import TrialEvent

_ENVIRONMENTS = {}
_BRIDGES = {}
_SAFE_SESSION = re.compile("^[A-Za-z0-9][A-Za-z0-9_.-]{0,220}__env$")
_CONTAINER_ID = re.compile("^[a-f0-9]{12,64}$")


COMPLETION_REASONS = frozenset(
    {
        "submitted",
        "inference_budget_exhausted",
        "inference_attempt_budget_exhausted",
        "project_request_budget_exhausted",
        "project_attempt_budget_exhausted",
        "project_spend_budget_exhausted",
        "walltime_exhausted",
    }
)


def _trusted_environment(env):
    return type(env) is BenchmarkDockerEnvironment


def _directory(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError("Trusted directory must not contain symlinks")
    path.mkdir(mode=448, parents=True, exist_ok=True)
    if (
        not path.is_dir()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 18
    ):
        raise ValueError("Trusted directory must be owned and privately writable")
    return path


@dataclass(frozen=True)
class _ActorStopReceipt:
    bridge_id: str
    episode_id: str
    container_id: str
    snapshot_id: str
    snapshot_sha256: str | None
    snapshot_error_code: str | None
    termination_reason: str
    raw_exit_code: int = 137


class NativeActorStop(NonZeroAgentExitCodeError):
    """Internal signal, accepted only with the lifecycle's identical receipt."""

    def __init__(self, receipt):
        super().__init__("Authoring stopped at submission or its configured limit")
        self.receipt = receipt


class NativeLifecycle:
    """Coordinate actor execution, stopping and immutable final submissions."""

    def __init__(
        self,
        controller,
        actor_base,
        *,
        deadline_seconds=1800,
        inference_limit=128,
        preview_limit=12,
        validation_limit=5,
        stop_timeout_seconds=120,
        bridge_id=None,
        tool_protocol="legacy",
    ):
        if version("harbor") != "0.24.0":
            raise RuntimeError("Native lifecycle requires pinned Harbor 0.24.0")
        if (
            not math.isfinite(stop_timeout_seconds)
            or not 0 < stop_timeout_seconds <= 600
        ):
            raise ValueError("Bounded Docker stop timeout required")
        if (
            not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
            or any(
                (
                    type(limit) is not int or limit < 0
                    for limit in (inference_limit, preview_limit, validation_limit)
                )
            )
        ):
            raise ValueError("Valid authoritative episode limits required")
        self.controller = controller
        from simready_bench.engine.tool_protocol import validate_tool_protocol

        self.tool_protocol = validate_tool_protocol(tool_protocol)
        self.actor_base = _directory(actor_base)
        if self.actor_base.is_relative_to(
            controller.state_dir
        ) or controller.state_dir.is_relative_to(self.actor_base):
            raise ValueError(
                "Actor files and trusted controller state must be separate"
            )
        self.bridge_id = str(UUID(bridge_id)) if bridge_id is not None else str(uuid4())
        self.services = None
        self._deferred_agents = []
        self.deadline_seconds = deadline_seconds
        self.inference_limit = inference_limit
        self.preview_limit = preview_limit
        self.validation_limit = validation_limit
        self.stop_timeout_seconds = stop_timeout_seconds
        self._episodes = {}
        self._stop_operations = {}
        self._watchers = {}
        self._loop = None
        self._controlled_stops = {}
        self._controlled_receipts = {}
        self._stop_attempts = {}
        self._stop_deadlines = {}

    def install(self):
        if self.bridge_id in _BRIDGES:
            raise RuntimeError("Bridge already installed")
        _BRIDGES[self.bridge_id] = self

    def uninstall(self):
        if any(
            not task.done()
            for task in (*self._watchers.values(), *self._stop_operations.values())
        ):
            raise RuntimeError("Cannot uninstall active lifecycle supervision")
        _BRIDGES.pop(self.bridge_id, None)
        for key in list(_ENVIRONMENTS):
            if key[0] == self.bridge_id:
                del _ENVIRONMENTS[key]

    def attach(self, job):
        for event, callback in (
            (TrialEvent.START, self.on_start),
            (TrialEvent.AGENT_START, self.on_agent_start),
            (TrialEvent.AGENT_END, self.on_agent_end),
            (TrialEvent.CANCEL, self.on_cancel),
            (TrialEvent.END, self.on_end),
        ):
            job.add_hook(event, callback)

    def _environment(self, event):
        env = _ENVIRONMENTS.get((self.bridge_id, event.trial_name + "__env"))
        if env is None or str(env.context_id) != str(event.trial_id):
            raise RuntimeError("Native environment identity does not match trial hook")
        return env

    async def on_start(self, event):
        self._loop = asyncio.get_running_loop()
        env = self._environment(event)
        episode_id = str(event.trial_id)
        await asyncio.to_thread(
            self.controller.register_episode,
            episode_id,
            env.actor_root,
            task_id=event.task_name,
            trial_id=episode_id,
            runner="harbor",
            deadline_seconds=self.deadline_seconds,
            inference_limit=self.inference_limit,
            preview_limit=self.preview_limit,
            validation_limit=self.validation_limit,
        )
        self._episodes[episode_id] = env
        env.episode_id = episode_id

    async def on_agent_start(self, event):
        env = self._environment(event)
        episode = await asyncio.to_thread(
            self.controller.start_episode, str(event.trial_id)
        )
        env.actor_phase = True
        if self.services is not None:
            await self.services.bind_episode(event, env)
        self._watchers[episode.episode_id] = asyncio.create_task(
            self._watch(episode.episode_id)
        )

    async def _watch(self, episode_id):
        while True:
            episode = await asyncio.to_thread(self.controller.get_episode, episode_id)
            if episode.status == "closed":
                await self.request_stop(
                    episode_id, episode.termination_reason or "closed"
                )
                return
            if episode.deadline_at is not None and time.time() >= episode.deadline_at:
                await self.request_stop(episode_id, "walltime_exhausted")
                return
            await asyncio.sleep(0.05)

    def _actor_receipt_matches(self, env, receipt):
        if (
            not _trusted_environment(env)
            or env.bridge is not self
            or _BRIDGES.get(self.bridge_id) is not self
            or (self._episodes.get(env.episode_id) is not env)
            or (self._controlled_receipts.get(env.episode_id) is not receipt)
        ):
            return False
        snapshot = env.final_snapshot
        episode = self.controller.get_episode(env.episode_id)
        with closing(self.controller._connect()) as db:
            row = db.execute(
                "SELECT e.final_snapshot_id,s.record FROM episodes e LEFT JOIN snapshots s ON s.snapshot_id=e.final_snapshot_id WHERE e.episode_id=?",
                (env.episode_id,),
            ).fetchone()
        if row is None or row["record"] is None or snapshot is None:
            return False
        if json.loads(row["record"]) != json.loads(json.dumps(asdict(snapshot))):
            return False
        audit = getattr(env, "_actor_stop_audit", None) or {}
        state = audit.get("state", {})
        proof = env.network_isolation_proof or {}
        kill_confirmed = audit.get("kill_return_code") == 0 or (
            audit.get("kill_return_code") is None
            and audit.get("kill_command_error")
            == {"error_type": "RuntimeError", "code": "compose_kill_timeout"}
            and state.get("Restarting") is False
        )
        return (
            receipt.bridge_id == self.bridge_id
            and receipt.episode_id == env.episode_id
            and (env.actor_stopped is True)
            and (snapshot is not None)
            and (episode.status == "closed")
            and (episode.termination_reason == receipt.termination_reason)
            and receipt.termination_reason in COMPLETION_REASONS
            and (
                row["final_snapshot_id"] == snapshot.snapshot_id == receipt.snapshot_id
            )
            and (snapshot.episode_id == env.episode_id)
            and (snapshot.sha256 == receipt.snapshot_sha256)
            and (snapshot.error_code == receipt.snapshot_error_code)
            and (
                audit.get("container_id")
                == proof.get("container_id")
                == receipt.container_id
            )
            and (audit.get("was_running") is True)
            and kill_confirmed
            and (state.get("Running") is False)
            and (state.get("Paused") is False)
            and (state.get("OOMKilled") is False)
            and (state.get("ExitCode") == 137)
        )

    def _stop_wait_timeout(self, env):
        timeout = (
            self.stop_timeout_seconds + self.controller.snapshot_timeout_seconds + 2
        )
        return timeout

    async def confirmed_actor_exit(self, env, return_code):
        """Await the existing verified stop and its durable final snapshot."""
        if (
            return_code != 137
            or not _trusted_environment(env)
            or env.bridge is not self
            or (self._episodes.get(env.episode_id) is not env)
        ):
            return None
        pending = self._controlled_stops.get(env.episode_id)
        if pending is None or pending[0] is not env:
            return None
        timeout = self._stop_wait_timeout(env)
        receipt = await asyncio.wait_for(asyncio.shield(pending[1]), timeout=timeout)
        if receipt is None or not self._actor_receipt_matches(env, receipt):
            return None
        record = asdict(receipt) | {
            "status": "confirmed_actor_stop",
            "actor_stopped": True,
            "state": env._actor_stop_audit["state"],
            "kill_return_code": env._actor_stop_audit["kill_return_code"],
        }
        raw = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
        directory = self.controller.state_dir / "native-environments"
        directory.mkdir(mode=448, exist_ok=True)
        path = directory / (
            hashlib.sha256((self.bridge_id + ":" + env.episode_id).encode()).hexdigest()
            + ".actor-stop.json"
        )
        try:
            with path.open("xb") as stream:
                stream.write(raw)
            path.chmod(384)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise RuntimeError("Confirmed submission stop receipt changed")
        return receipt

    def accept_actor_stop(self, agent, env, signal):
        """Finish the bound agent only after its verified submission or limit stop."""
        return (
            type(signal) is NativeActorStop
            and agent in self._deferred_agents
            and (agent.native_bridge_id == self.bridge_id)
            and (agent._native_bound_episode == env.episode_id)
            and self._actor_receipt_matches(env, signal.receipt)
        )

    def completed_actor_stop(self, env):
        receipt = self._controlled_receipts.get(env.episode_id)
        return receipt is not None and self._actor_receipt_matches(env, receipt)

    async def request_stop(self, episode_id, reason):
        """Async gateway stop callback. True means stopped and final capture durable."""
        if episode_id not in self._episodes:
            raise RuntimeError("No native actor registered for episode")
        operation = self._stop_operations.get(episode_id)
        failed = operation is not None and operation.done() and (
            operation.cancelled() or operation.exception() is not None
        )
        if operation is None or failed:
            attempt = self._stop_attempts.get(episode_id, 0) + 1
            deadline = self._stop_deadlines.setdefault(
                episode_id, time.monotonic() + 3 * self._stop_wait_timeout(self._episodes[episode_id])
            )
            if attempt > 3 or time.monotonic() >= deadline:
                raise RuntimeError("Native stop recovery allowance exhausted")
            self._stop_attempts[episode_id] = attempt
            operation = asyncio.create_task(self._stop_transition(episode_id, reason, attempt))
            self._stop_operations[episode_id] = operation
        # Stopping the actor also closes its outstanding HTTP connections. Their
        # handlers may disappear, but the one owned stop and capture must finish.
        return await asyncio.shield(operation)

    async def _stop_transition(self, episode_id, reason, attempt):
        """Retain each bounded transition and resume only unfinished phases."""
        status, error_type = "failed", None
        try:
            timeout = min(self._stop_wait_timeout(self._episodes[episode_id]),
                          self._stop_deadlines[episode_id] - time.monotonic())
            async with asyncio.timeout(max(0.001, timeout)):
                result = await self._finish_stop(episode_id, reason)
            status = "completed"
            return result
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            env = self._episodes[episode_id]
            snapshot = env.final_snapshot
            record = {
                "schema": "simready-native-stop-transition-v1",
                "bridge_id": self.bridge_id, "episode_id": episode_id,
                "attempt": attempt, "status": status, "error_type": error_type,
                "actor_stopped": env.actor_stopped,
                "snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
                "snapshot_sha256": snapshot.sha256 if snapshot is not None else None,
                "stop_progress": getattr(env, "_actor_stop_progress", None),
                "termination_reason": self.controller.get_episode(episode_id).termination_reason,
            }
            directory = self.controller.state_dir / "native-stop-transitions"
            directory.mkdir(mode=0o700, exist_ok=True)
            name = hashlib.sha256((self.bridge_id + ":" + episode_id).encode()).hexdigest()
            path = directory / (name + "." + str(attempt).zfill(3) + ".json")
            with path.open("x") as stream:
                json.dump(record, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o600)

    async def _finish_stop(self, episode_id, reason):
        env = self._episodes.get(episode_id)
        if env is None:
            raise RuntimeError("No native actor registered for episode")
        episode = await asyncio.to_thread(
            self.controller.close_episode, episode_id, reason
        )
        if self.services is not None:
            self.services.revoke_episode(episode_id)
        if env.final_snapshot is not None:
            if self.services is not None:
                await self.services.ensure_verifier(episode_id)
            return True
        pending = None
        receipt = None
        if episode.termination_reason in COMPLETION_REASONS:
            pending = asyncio.get_running_loop().create_future()
            self._controlled_stops[episode_id] = (env, pending)
        try:
            await env.stop_actor_writes()
            snapshot = await asyncio.to_thread(
                self.controller.finalize_submission,
                episode_id,
                reason=reason,
                **{"candidate_error": env.candidate_import_error}
                if getattr(env, "candidate_import_error", None)
                else {},
            )
            env.final_snapshot = snapshot
            env.actor_phase = False
            if pending is not None:
                audit = getattr(env, "_actor_stop_audit", None) or {}
                receipt = _ActorStopReceipt(
                    self.bridge_id,
                    episode_id,
                    audit.get("container_id"),
                    snapshot.snapshot_id,
                    snapshot.sha256,
                    snapshot.error_code,
                    episode.termination_reason,
                )
                self._controlled_receipts[episode_id] = receipt
            if self.services is not None:
                await self.services.ensure_verifier(episode_id)
            return True
        finally:
            if pending is not None:
                pending.set_result(receipt)

    def stop_actor_sync(self, episode_id, reason):
        """Thread-only ActorTools callback; never block the lifecycle event loop."""
        if self._loop is None:
            raise RuntimeError("Native lifecycle loop not running")
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self._loop:
            raise RuntimeError(
                "Synchronous stop adapter must run outside native event loop"
            )
        future = asyncio.run_coroutine_threadsafe(
            self.request_stop(episode_id, reason), self._loop
        )
        return future.result(
            timeout=self._stop_wait_timeout(self._episodes.get(episode_id))
        )

    async def on_agent_end(self, event):
        episode = self.controller.get_episode(str(event.trial_id))
        if episode.deadline_at is not None and time.time() >= episode.deadline_at:
            reason = "walltime_exhausted"
        elif episode.inference_calls_used >= episode.inference_limit:
            reason = "inference_budget_exhausted"
        else:
            reason = "agent_completed"
        await self.request_stop(str(event.trial_id), reason)

    async def on_cancel(self, event):
        if str(event.trial_id) in self._episodes:
            await self.request_stop(str(event.trial_id), "cancelled")

    async def on_end(self, event):
        episode_id = str(event.trial_id)
        if episode_id in self._episodes:
            await self.request_stop(episode_id, "trial_ended")
        watcher = self._watchers.pop(episode_id, None)
        if watcher is not None and watcher is not asyncio.current_task():
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
        if self.services is not None:
            await self.services.close_episode(episode_id)


class BenchmarkDockerEnvironment(DockerEnvironment):
    """Supported native environment extension; actor and verifier roles separate."""

    def __init__(self, *args, simready_bridge_id, **kwargs):
        bridge = _BRIDGES.get(simready_bridge_id)
        if bridge is None:
            raise RuntimeError("Trusted native lifecycle bridge must be installed")
        self.bridge = bridge
        self.actor_root = None
        self.episode_id = None
        self.actor_phase = False
        self.actor_stopped = False
        self.actor_start_attempted = False
        self.final_snapshot = None
        self._actor_stop_audit = None
        session = kwargs.get("session_id")
        if not isinstance(session, str):
            raise ValueError("Explicit native session identity required")
        actor = session.endswith("__env") and "__verifier__" not in session
        if actor:
            if not _SAFE_SESSION.fullmatch(session):
                raise ValueError("Unsafe actor session")
            retain_owned = False
            if (
                kwargs.get("extra_docker_compose")
                or kwargs.get("stream")
                or (kwargs.get("keep_containers") and (not retain_owned))
            ):
                raise ValueError(
                    "Actor overlays, stream and unbound retained containers are unsupported"
                )
            paths = kwargs["trial_paths"]
            allowed = {
                "/logs/agent": paths.agent_dir,
                "/logs/artifacts": paths.host_artifact_path(
                    MAIN_SERVICE_NAME, "/logs/artifacts"
                ),
            }
            mounts = []
            for mount in kwargs.get("mounts") or []:
                target = mount.get("target")
                source = Path(mount.get("source", "")).absolute()
                if (
                    target == "/logs/verifier"
                    and source == paths.verifier_dir.absolute()
                ):
                    continue
                if (
                    target not in allowed
                    or source != allowed[target].absolute()
                    or mount.get("type") != "bind"
                ):
                    raise ValueError(
                        "Only native actor log mounts and fixed output mount are permitted"
                    )
                item = dict(mount)
                if target == "/logs/artifacts":
                    item["read_only"] = True
                mounts.append(item)
            root = bridge.actor_base / session
            root.mkdir(mode=448, exist_ok=False)
            (root / "output").mkdir(mode=511)
            (root / "output").chmod(511)
            self.actor_root = root
            mounts.append(
                {
                    "type": "bind",
                    "source": str(root / "output"),
                    "target": "/output",
                    "bind": {"create_host_path": False},
                }
            )
            if bridge.services is not None:
                mounts.extend(
                    bridge.services.actor_mounts_for_directory(
                        kwargs["environment_dir"], root
                    )
                )
            kwargs["mounts"] = mounts
        elif "__verifier__" not in session:
            raise ValueError("Unknown native environment role")
        else:
            kwargs["keep_containers"] = False
        if not actor and bridge.services is not None:
            kwargs["mounts"] = list(
                kwargs.get("mounts") or []
            ) + bridge.services.verifier_mounts(kwargs["trial_paths"])
        super().__init__(*args, **kwargs)
        if (
            self._uses_compose
            or self._is_windows_container
            or self.extra_docker_compose_paths
        ):
            raise ValueError(
                "Native Unix transport requires a prebuilt Linux image without task overlays"
            )
        self._enable_egress_control = False
        overlay_dir = bridge.controller.state_dir / "native-environments"
        overlay_dir.mkdir(mode=448, exist_ok=True)
        overlay = overlay_dir / (
            hashlib.sha256(session.encode()).hexdigest() + ".network-none.json"
        )
        with overlay.open("x") as stream:
            json.dump({"services": {"main": {"network_mode": "none"}}}, stream)
        overlay.chmod(384)
        self.extra_docker_compose_paths.append(overlay)
        self.network_isolation_proof = None
        if actor:
            key = (simready_bridge_id, session)
            if key in _ENVIRONMENTS:
                raise ValueError("Duplicate native actor session")
            _ENVIRONMENTS[key] = self

    @property
    def capabilities(self):
        capabilities = super().capabilities
        return capabilities.model_copy(
            update={
                name: False
                for name in type(capabilities).model_fields
                if name.startswith("network_allowlist")
            }
            | {
                "disable_internet": True,
                "dynamic_network_policy": False,
                "stream": False,
                "gpus": False,
                "windows": False,
            }
        )

    async def start(self, force_build):
        if force_build or not self.task_env_config.docker_image:
            raise RuntimeError(
                "Native transport requires an existing prebuilt image"
            )
        if self.network_policy.network_mode.value != "no-network":
            raise RuntimeError(
                "Native Unix transport requires no-network for both roles"
            )
        self.actor_start_attempted = self.actor_root is not None
        await super().start(force_build)
        result = await self._run_docker_compose_command(
            ["ps", "--all", "--quiet", MAIN_SERVICE_NAME], timeout_sec=2
        )
        identifiers = (result.stdout or "").split()
        if len(identifiers) != 1 or not _CONTAINER_ID.fullmatch(identifiers[0]):
            raise RuntimeError("Exactly one actual native role container required")
        identifier = identifiers[0]
        network_mode = (
            (
                await self._engine(
                    "inspect", "--format", "{{.HostConfig.NetworkMode}}", identifier
                )
            )
            .decode()
            .strip()
        )
        if network_mode != "none":
            await self._run_docker_compose_command(
                ["kill", "-s", "SIGKILL", MAIN_SERVICE_NAME], check=False, timeout_sec=2
            )
            raise RuntimeError(
                "Actual native container is not isolated with Docker network none"
            )
        image_id = (
            (await self._engine("inspect", "--format", "{{.Image}}", identifier))
            .decode()
            .strip()
        )
        self.network_isolation_proof = {
            "container_id": identifier,
            "network_mode": network_mode,
            "image_id": image_id,
            "session_id": self.session_id,
            "role": "actor" if self.actor_root is not None else "verifier",
        }
        proof_path = (
            self.bridge.controller.state_dir
            / "native-environments"
            / (
                hashlib.sha256(self.session_id.encode()).hexdigest()
                + ".actual-isolation.json"
            )
        )
        with proof_path.open("x") as stream:
            json.dump(self.network_isolation_proof, stream, sort_keys=True)
        proof_path.chmod(384)
        if self.actor_root is not None and self.bridge.services is not None:
            search = await super().exec(
                "command -v rg >/dev/null && rg --version >/dev/null",
                user=str(os.getuid()),
                timeout_sec=5,
            )
            if search.return_code != 0:
                raise RuntimeError(
                    "Authoring image must provide ripgrep for complete diagnostic files"
                )
            command = "python3 -m simready_bench.engine.actor_relay --unix-socket /run/simready-services/service.sock --port 32190 --ready-file /tmp/simready-relay.ready >/logs/agent/relay.log 2>&1 </dev/null &"
            await super().exec(command, user=str(os.getuid()), timeout_sec=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = await super().exec(
                    "test -s /tmp/simready-relay.ready",
                    user=str(os.getuid()),
                    timeout_sec=1,
                )
                if result.return_code == 0:
                    break
                await asyncio.sleep(0.05)
            else:
                raise RuntimeError("Native actor Unix relay did not become ready")

    async def _engine(self, *arguments):
        process = await asyncio.create_subprocess_exec(
            *type(self)._engine_cmd(*arguments),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=min(30, self.bridge.stop_timeout_seconds)
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode != 0 or len(stdout) > 65536:
            raise RuntimeError("Docker actor state inspection failed")
        return stdout

    async def stop_actor_writes(self):
        if self.actor_root is None:
            raise RuntimeError("Cannot stop verifier as actor")
        if self.actor_stopped:
            return
        if not self.actor_start_attempted:
            self.actor_stopped = True
            return
        async with asyncio.timeout(self.bridge.stop_timeout_seconds):
            result = await self._run_docker_compose_command(
                ["ps", "--all", "--quiet", MAIN_SERVICE_NAME], timeout_sec=30
            )
            identifiers = (result.stdout or "").split()
            if len(identifiers) != 1 or not _CONTAINER_ID.fullmatch(identifiers[0]):
                raise RuntimeError("Exactly one actual native actor container required")
            identifier = identifiers[0]
            directory = self.bridge.controller.state_dir / "native-environments"
            directory.mkdir(mode=0o700, exist_ok=True)
            prefix = hashlib.sha256((self.bridge.bridge_id + ":" + self.episode_id).encode()).hexdigest()
            progress_path = directory / (prefix + ".actor-stop-progress.json")
            progress = getattr(self, "_actor_stop_progress", None)
            if progress is None and progress_path.is_file():
                progress = self._actor_stop_progress = json.loads(progress_path.read_text())
            if progress is not None and progress["container_id"] != identifier:
                raise RuntimeError("Native stop recovery container changed")

            def save_progress():
                temporary = progress_path.with_suffix(".tmp")
                with temporary.open("w") as stream:
                    json.dump(progress, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(0o600)
                temporary.replace(progress_path)
            try:
                before = json.loads(
                    await self._engine(
                        "inspect", "--format", "{{json .State}}", identifier
                    )
                )
            except (RuntimeError, ValueError, TypeError):
                before = {}
            if progress is None:
                progress = self._actor_stop_progress = {
                    "container_id": identifier, "was_running": before.get("Running") is True,
                    "kill_return_code": None, "kill_command_error": None,
                }
                save_progress()
            # A previous kill may have succeeded before inspection failed. Observe
            # the same container again before repeating its destructive operation.
            if before.get("Running") is not False or before.get("Paused") is not False:
                try:
                    killed = await self._run_docker_compose_command(
                        ["kill", "-s", "SIGKILL", MAIN_SERVICE_NAME],
                        check=False, timeout_sec=30,
                    )
                    progress["kill_return_code"] = killed.return_code
                    save_progress()
                except RuntimeError as exc:
                    if (
                        type(exc) is not RuntimeError
                        or str(exc) != "Command timed out after 30 seconds"
                        or (self.network_isolation_proof or {}).get("container_id") != identifier
                    ):
                        raise
                    progress["kill_command_error"] = {
                        "error_type": "RuntimeError", "code": "compose_kill_timeout",
                    }
                    save_progress()
            kill_error = progress["kill_command_error"]
            state = json.loads(
                await self._engine("inspect", "--format", "{{json .State}}", identifier)
            )
            if (
                state.get("Running") is not False
                or state.get("Paused") is not False
                or (kill_error is not None and state.get("Restarting") is not False)
            ):
                raise RuntimeError(
                    "Actor is not confirmed stopped; final capture refused"
                )
        self._actor_stop_audit = {
            "container_id": identifier,
            "was_running": progress["was_running"],
            "kill_return_code": progress["kill_return_code"],
            "state": state,
        }
        if kill_error is not None:
            self._actor_stop_audit["kill_command_error"] = kill_error
        directory = self.bridge.controller.state_dir / "native-environments"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / (
            hashlib.sha256(
                (self.bridge.bridge_id + ":" + self.episode_id).encode()
            ).hexdigest()
            + ".actor-stop-audit.json"
        )
        raw = (
            json.dumps(self._actor_stop_audit, sort_keys=True, indent=2) + "\n"
        ).encode()
        if path.exists():
            if path.read_bytes() != raw:
                raise RuntimeError("Native stop audit identity changed")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
        path.chmod(0o600)
        self.actor_stopped = True

    async def exec(self, *args, **kwargs):
        if self.actor_stopped and self.actor_root is not None:
            receipt = await self.bridge.confirmed_actor_exit(self, 137)
            if receipt is not None:
                raise NativeActorStop(receipt)
            raise NonZeroAgentExitCodeError("Benchmark actor phase stopped")
        if self.actor_phase and self.bridge.services is not None:
            kwargs["env"] = {
                **(kwargs.get("env") or {}),
                **self.bridge.services.episode_environment(self.episode_id),
            }
        if self.actor_root is None:
            fixed = "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"
            interpreted = (
                "(/bin/sh /tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"
            )
            if kwargs.get("command") == fixed:
                kwargs["command"] = interpreted
            elif args and args[0] == fixed:
                args = (interpreted, *args[1:])
        result = await super().exec(*args, **kwargs)
        if self.actor_root is not None and result.return_code == 137:
            receipt = await self.bridge.confirmed_actor_exit(self, result.return_code)
            if receipt is not None:
                raise NativeActorStop(receipt)
        if self.actor_stopped and self.actor_root is not None:
            receipt = await self.bridge.confirmed_actor_exit(self, 137)
            if receipt is not None:
                raise NativeActorStop(receipt)
            raise NonZeroAgentExitCodeError("Benchmark actor phase stopped")
        return result

    async def prepare_logs_for_host(self):
        if self.actor_root is not None and self.actor_stopped:
            return
        await super().prepare_logs_for_host()

    async def service_is_dir(self, path, *, service=None, user=None):
        if (
            self.actor_root is not None
            and path == "/output"
            and (service in (None, MAIN_SERVICE_NAME))
        ):
            return True
        return await super().service_is_dir(path, service=service, user=user)

    async def service_download_dir(self, source_dir, target_dir, *, service=None):
        if (
            self.actor_root is not None
            and source_dir == "/output"
            and (service in (None, MAIN_SERVICE_NAME))
        ):
            snapshot = self.final_snapshot
            if not self.actor_stopped or snapshot is None:
                raise RuntimeError("Final native candidate was not frozen")
            target = Path(target_dir)
            if target.is_symlink() or any(target.iterdir()):
                raise RuntimeError("Native candidate destination must be empty")
            if snapshot.valid:
                await asyncio.to_thread(self._export_snapshot, snapshot, target)
            return
        return await super().service_download_dir(
            source_dir, target_dir, service=service
        )

    @staticmethod
    def _export_snapshot(snapshot, target):
        source = Path(snapshot.path)
        for record in snapshot.files:
            relative = Path(record.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Invalid trusted snapshot manifest")
            original = source / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record.sha256:
                    raise RuntimeError("Frozen snapshot digest mismatch")
            shutil.copyfile(original, destination)

    async def stop(self, delete):
        if self.actor_root is not None and (not self.actor_start_attempted):
            if self.episode_id is not None:
                await self.bridge.request_stop(
                    self.episode_id, "environment_not_started"
                )
            return
        try:
            if self.actor_root is not None and self.episode_id is not None:
                await self.bridge.request_stop(self.episode_id, "environment_stopped")
        finally:
            await super().stop(delete)


@dataclass(frozen=True)
class NativeActorStopAdapter:
    """Exact trusted ActorTools adapter; no actor-supplied callback or command."""

    lifecycle: NativeLifecycle
    kind: ClassVar[str] = "stop_actor"

    def __post_init__(self):
        if type(self.lifecycle) is not NativeLifecycle:
            raise ValueError("Exact native lifecycle coordinator required")

    @property
    def controller(self):
        return self.lifecycle.controller

    def __call__(self, episode_id, reason):
        return self.lifecycle.stop_actor_sync(episode_id, reason)


def register_deferred_agent(bridge_id, agent):
    bridge = _BRIDGES.get(bridge_id)
    if bridge is None or bridge.services is None:
        raise RuntimeError("Installed native service coordinator required")
    from simready_bench.engine.pi import PiAgent
    from simready_bench.harnesses import BenchmarkPi
    from simready_bench.harbor_agents import is_scoped_native_agent

    if (
        type(agent) not in (PiAgent, BenchmarkPi)
        and (not is_scoped_native_agent(agent))
        or agent in bridge._deferred_agents
    ):
        raise ValueError("Exactly one native agent must bind to a trial")
    bridge._deferred_agents.append(agent)
