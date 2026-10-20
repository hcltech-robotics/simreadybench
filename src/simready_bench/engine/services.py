"""Trusted native service transport over a private mounted Unix socket."""

from __future__ import annotations
import asyncio
import json
import time
from aiohttp import web
from simready_bench.engine.inference_gateway import InferenceGateway
from simready_bench.engine.tools import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, _error


class NativeServiceTransport(InferenceGateway):
    """Same bounded HTTP protocol for scoped inference and real ActorTools.

    The raw actor loopback relay does not route to any network origin. Only this
    trusted server owns the upstream provider and calls the authoritative tools.
    """

    def __init__(self, controller, config, *, tools, on_stop_requested=None):
        if tools.controller is not controller:
            raise ValueError("Native services must share the authoritative controller")
        super().__init__(controller, config, on_stop_requested=on_stop_requested)
        self.tools = tools

    async def handle(self, request):
        if not request.path.startswith("/v1/tools/"):
            return await super().handle(request)
        token = request.headers.get("Authorization", "")
        token = token[7:] if token.startswith("Bearer ") else ""
        if self.tools._binding(token) is None:
            return web.Response(
                status=401,
                body=_error("unauthorized")
                .model_dump_json(by_alias=True, exclude_none=True)
                .encode(),
                content_type="application/json",
            )
        if request.method != "POST" or request.query_string:
            return web.Response(
                status=400,
                body=_error("invalid_arguments")
                .model_dump_json(by_alias=True, exclude_none=True)
                .encode(),
                content_type="application/json",
            )
        operation = request.path[len("/v1/tools/") :]
        if "/" in operation:
            operation = "unknown_operation"
        try:
            async with asyncio.timeout(5):
                if (
                    request.content_length is None
                    or request.content_length > MAX_REQUEST_BYTES
                ):
                    raise ValueError("Bounded fixed-length tool request required")
                raw = await request.read()
                if len(raw) > MAX_REQUEST_BYTES:
                    raise ValueError("Tool request too large")
                arguments = json.loads(raw)
            binding = self.tools._binding(token)
            episode = self.controller.get_episode(binding.episode_id)
            remaining = max(
                0.001, (episode.deadline_at or time.time() + 5) - time.time()
            )
            async with asyncio.timeout(
                min(self.config.request_timeout_seconds, remaining + 1)
            ):
                async with self._semaphore:
                    if operation == "tool_definitions":
                        result = (
                            await asyncio.to_thread(self.tools.describe, token)
                            if arguments == {}
                            else _error("invalid_arguments")
                        )
                    else:
                        result = await asyncio.to_thread(
                            self.tools.invoke, token, operation, arguments
                        )
            data = result.model_dump_json(by_alias=True, exclude_none=True).encode()
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("Tool response too large")
            response = web.StreamResponse(
                headers={
                    "Content-Type": "application/json",
                    "Cache-Control": "no-store",
                }
            )
            try:
                async with asyncio.timeout(5):
                    await response.prepare(request)
                    for offset in range(0, len(data), 65536):
                        await response.write(data[offset : offset + 65536])
                    await response.write_eof()
            except (asyncio.TimeoutError, ConnectionError):
                if request.transport is not None:
                    request.transport.abort()
            return response
        except (ValueError, UnicodeError):
            status = 400
            result = _error("invalid_arguments")
        except asyncio.TimeoutError:
            status = 504
            result = _error("tool_deadline_exceeded")
        return web.Response(
            status=status,
            body=result.model_dump_json(by_alias=True, exclude_none=True).encode(),
            content_type="application/json",
        )


from pathlib import Path
import shutil
import tempfile


class NativeServices:
    """Actual per-job IPC services and per-trial tokens, outside actor sandboxes."""

    def __init__(
        self,
        lifecycle,
        config,
        *,
        task_inputs,
        evaluator=None,
        verifier_timeout_seconds=600,
    ):
        self.lifecycle = lifecycle
        self.config = config
        self.inputs = {str(record.task_path): record for record in task_inputs}
        if not self.inputs or len(self.inputs) != len(task_inputs):
            raise ValueError("Unique trusted task bindings required")
        if not 0 < verifier_timeout_seconds <= 600:
            raise ValueError("Bounded final verifier deadline required")
        self.verifier_timeout_seconds = verifier_timeout_seconds
        self.evaluator = evaluator
        self.socket_dir = None
        self.runner = None
        self.tokens = {}
        self.verifiers = {}
        lifecycle.services = self

    def public_input(self, environment_dir):
        key = str(Path(environment_dir).absolute().parent)
        if key not in self.inputs:
            raise RuntimeError("Native environment has no trusted public task binding")
        return self.inputs[key]

    async def start(self):
        if self.runner is not None:
            raise RuntimeError("Native services already started")
        self.socket_dir = Path(tempfile.mkdtemp(prefix="simready-actor-"))
        self.socket_dir.chmod(448)
        self.runner = self.transport.runner()
        try:
            await self.runner.setup()
            await self.runner.unix_site(self.socket_dir / "service.sock").start()
        except BaseException:
            await self.runner.cleanup()
            self.runner = None
            raise

    async def bind_episode(self, event, env):
        if self.runner is None:
            raise RuntimeError("Native services are not running")
        episode_id = str(event.trial_id)
        record = self.public_input(env.environment_dir)
        token = self.transport.issue_episode_token(episode_id)
        self.tools.bind_episode(
            episode_id,
            token,
            record.thumbnail,
            record.public_text,
            public_contract=record.public_repair_contract,
        )
        self.tokens[episode_id] = token
        agents = [
            agent
            for agent in self.lifecycle._deferred_agents
            if str(agent.context_id) == episode_id
        ]
        if len(agents) > 1:
            raise RuntimeError("Multiple native agents match one trial")
        if (
            event.config.agent.import_path
            in (
                "simready_bench.engine.pi:PiAgent",
                "simready_bench.harbor_agents:ScopedNativeAgent",
            )
            and len(agents) != 1
        ):
            raise RuntimeError(
                "Pi native episode was not registered through deferred binding"
            )
        for agent in agents:
            agent.bind_native_episode(
                self.lifecycle.bridge_id, episode_id, token, "http://127.0.0.1:32190"
            )

    def episode_environment(self, episode_id):
        token = self.tokens.get(episode_id)
        if token is None:
            raise RuntimeError("Native episode service token is not bound")
        result = {
            "SIMREADY_EPISODE_TOKEN": token,
            "OPENAI_API_KEY": token,
            "OPENAI_BASE_URL": "http://127.0.0.1:32190/v1",
        }
        for agent in self.lifecycle._deferred_agents:
            if str(agent.context_id) == str(episode_id):
                from simready_bench.harbor_agents import (
                    is_scoped_native_agent,
                    deferred_environment,
                )

                if is_scoped_native_agent(agent):
                    for key in deferred_environment(type(agent)._native_class):
                        result[key] = agent._extra_env[key]
        return result

    def revoke_episode(self, episode_id):
        self.transport.revoke_episode(episode_id)
        self.tokens.pop(episode_id, None)
        with self.tools._guard:
            self.tools._bindings.pop(episode_id, None)

    def public_client_mounts(self, actor_root):
        directory = Path(actor_root) / "public-client"
        package = directory / "simready_bench"
        engine = package / "engine"
        engine.mkdir(parents=True, mode=448)
        (package / "__init__.py").write_text("")
        (engine / "__init__.py").write_text("")
        for name in ("actor_mcp.py", "actor_relay.py"):
            source = Path(__file__).with_name(name)
            if (
                source.is_symlink()
                or not source.is_file()
                or source.stat().st_size > 131072
            ):
                raise RuntimeError("Bounded public actor client required")
            shutil.copyfile(source, engine / name)
        for path in directory.rglob("*"):
            path.chmod(365 if path.is_dir() else 292)
        directory.chmod(365)
        return [
            {
                "type": "bind",
                "source": str(directory),
                "target": "/opt/simready/runtime",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        ]

    def verifier_mounts(self, trial_paths):
        envs = [
            env
            for env in self.lifecycle._episodes.values()
            if env.trial_paths.trial_dir == trial_paths.trial_dir
        ]
        if len(envs) != 1:
            raise RuntimeError("Private verifier must resolve one exact native trial")
        episode_id = envs[0].episode_id
        if episode_id not in self.verifiers:
            raise RuntimeError("Private frozen-snapshot verifier service not ready")
        directory, _ = self.verifiers[episode_id]
        return [
            {
                "type": "bind",
                "source": str(directory),
                "target": "/run/simready-verifier",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        ]

    async def close_episode(self, episode_id):
        self.revoke_episode(episode_id)
        record = self.verifiers.pop(episode_id, None)
        if record is not None:
            directory, runner = record
            await runner.cleanup()
            shutil.rmtree(directory)

    async def close(self):
        for episode_id in list(self.tokens):
            self.revoke_episode(episode_id)
        for directory, runner in self.verifiers.values():
            await runner.cleanup()
            shutil.rmtree(directory)
        self.verifiers.clear()
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
        if self.socket_dir:
            shutil.rmtree(self.socket_dir)
            self.socket_dir = None
