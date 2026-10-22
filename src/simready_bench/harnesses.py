"Configure installed Harbor harnesses and mount the selected skills."

from __future__ import annotations
import hashlib, json, shlex
from pathlib import Path, PurePosixPath
from simready_bench.engine.pi import PiAgent, DEFERRED_TOKEN
from simready_bench.engine.native_output import native_output, NATIVE_OUTPUT_POLICY


def _prompt(value):
    if not isinstance(value, str) or len(value) > 65536 or "\x00" in value:
        raise ValueError("Bounded literal system_prompt required")
    return value


class BenchmarkPi(PiAgent):
    """Pi 0.85.1 supports --system-prompt; its run/setup are inherited intact."""

    def __init__(self, *args, system_prompt="", retry_owner="harness", **kwargs):
        if retry_owner not in ("harness", "relay"):
            raise ValueError("Explicit Pi retry owner required")
        self._retry_owner = retry_owner
        self._system_prompt = _prompt(system_prompt)
        super().__init__(*args, **kwargs)

    @classmethod
    def preflight(cls, kwargs=None, env=None):
        options = dict(kwargs or {})
        _prompt(options.pop("system_prompt", ""))
        if options.pop("retry_owner", "harness") not in ("harness", "relay"):
            raise ValueError("Explicit Pi retry owner required")
        return super().preflight(options, env)

    async def _ensure_config_dir(self, environment):
        await super()._ensure_config_dir(environment)
        if self._retry_owner == "relay":
            from harbor.agents.installed.pi import _REMOTE_PI_CONFIG_DIR

            remote = (_REMOTE_PI_CONFIG_DIR / "settings.json").as_posix()
            await self._upload_config_text(
                environment,
                content=json.dumps(
                    {
                        "retry": {
                            "enabled": False,
                            "maxRetries": 0,
                            "provider": {"maxRetries": 0},
                        }
                    }
                )
                + "\n",
                remote_path=remote,
                filename="settings.json",
            )
            await self.exec_as_agent(
                environment, command="chmod 600 " + shlex.quote(remote)
            )

    def _prompt_cli_flags(self):
        return (
            "--system-prompt " + shlex.quote(self._system_prompt)
            if self._system_prompt
            else ""
        )

    def build_cli_flags(self):
        return " ".join(
            (x for x in (super().build_cli_flags(), self._prompt_cli_flags()) if x)
        )


def actor_public_packet(condition):
    """Explicit public projection; no reference records, paths or providers."""
    settings = condition["settings"]
    return {
        "schema": "simready-bench-actor-configuration-v1",
        "condition_id": condition["condition_id"],
        "asset_id": condition["asset_id"],
        "harness": condition["harness"]["adapter"],
        "system_prompt": _prompt(settings.get("system_prompt", "")),
        "limits": json.loads(json.dumps(settings["budgets"], allow_nan=False)),
        "skills_sha256": condition.get("skill_set", {}).get("sha256"),
    }


def _relative(value):
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or path.is_absolute()
        or (not path.parts)
        or any((p in ("", ".", "..") for p in path.parts))
        or ("\\" in value)
    ):
        raise ValueError("Confined relative skill path required")
    return path


def stage_skills(condition, output):
    """Stage selected skills for Harbor and mount their verified files read-only."""
    records = condition.get("skill_set", {}).get("closure", [])
    if not records:
        return []
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve skill staging")
    pending = []
    seen = set()
    total = 0
    for record in records:
        relative = _relative(record["path"])
        source = Path(record["absolute_path"]).absolute()
        if relative.as_posix() in seen:
            raise ValueError("Duplicate skill path")
        seen.add(relative.as_posix())
        total += record["size_bytes"]
        if total > 16 * 1024 * 1024:
            raise ValueError("Bounded public skill closure required")
        if (
            source.resolve() != source
            or not source.is_file()
            or source.stat().st_nlink != 1
        ):
            raise ValueError("Regular nonlinked skill file required")
        raw = source.read_bytes()
        if (
            len(raw) != record["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("Skill source bytes changed")
        pending.append((relative, raw))
    output.mkdir(parents=True, mode=493)
    for relative, raw in pending:
        target = output.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=493)
        target.write_bytes(raw)
        target.chmod(292)
    roots = sorted({p.parent for p in output.rglob("SKILL.md")})
    if len({p.name for p in roots}) != len(roots):
        raise ValueError("Duplicate leaf skill names would collapse in Harbor")
    if not roots:
        raise ValueError("Each selected skill closure must include SKILL.md")
    for directory in sorted((p for p in output.rglob("*") if p.is_dir()), reverse=True):
        directory.chmod(365)
    output.chmod(365)
    return roots


def agent_config(condition, bridge_id, skill_dirs):
    harness = condition["harness"]["adapter"]
    model = condition["model"]
    settings = condition["settings"]
    prompt = _prompt(settings.get("system_prompt", ""))
    if harness == "harbor":
        from simready_bench.harbor_agents import (
            resolve_native_class,
            validate_native_kwargs,
            deferred_environment,
        )

        name, cls = resolve_native_class(agent=condition["harness"]["native_agent"])
        options = validate_native_kwargs(
            name, cls, condition["harness"].get("kwargs", {})
        )
        generation = settings.get("generation", {})
        limits = dict(settings["budgets"])
        limits["max_output_tokens"] = generation.get(
            "max_output_tokens", limits["max_output_tokens"]
        )
        for key in ("max_tokens", "max_output_tokens"):
            if key in cls.options_model.model_fields:
                if key in options and options[key] > limits["max_output_tokens"]:
                    raise ValueError("Native output setting exceeds condition cap")
                options.setdefault(key, limits["max_output_tokens"])
        if any((generation.get(k) is not None for k in ("temperature", "top_p"))):
            raise ValueError(
                "Native harness does not apply temperature/top_p overrides"
            )
        effort = generation.get("reasoning_effort")
        if (
            effort not in (None, "off", "none")
            and name != "opencode"
            and ("reasoning_effort" not in cls.options_model.model_fields)
        ):
            raise ValueError(
                "Native harness does not expose a reasoning_effort setting"
            )
        if (
            effort not in (None, "off", "none")
            and "reasoning_effort" in cls.options_model.model_fields
        ):
            options.setdefault("reasoning_effort", effort)
        return {
            "import_path": "simready_bench.harbor_agents:ScopedNativeAgent",
            "model_name": "openai/" + model["model_id"],
            "skills": [str(Path(p).absolute()) for p in skill_dirs],
            "mcp_servers": [
                {
                    "name": "benchmark",
                    "transport": "stdio",
                    "command": "python3",
                    "args": ["-m", "simready_bench.engine.actor_mcp"],
                }
            ],
            "override_timeout_sec": settings["budgets"]["agent_seconds"],
            "env": deferred_environment(cls),
            "kwargs": {
                "native_agent": name,
                "native_kwargs": options,
                "native_bridge_id": bridge_id,
                "native_api_style": model["adapter"],
                "native_model_metadata": model["metadata"],
                "native_limits": limits,
                "native_system_prompt": prompt,
                "native_generation": generation,
            },
        }
    if harness != "pi" or model["adapter"] not in (
        "openai-responses",
        "openai-completions",
    ):
        raise ValueError("Unsupported harness/model adapter pair")
    generation = settings.get("generation", {})
    if any((generation.get(k) is not None for k in ("temperature", "top_p"))):
        raise ValueError(
            "Pi does not support temperature or top_p overrides"
        )
    native_options = condition["harness"].get("kwargs", {})
    from simready_bench.capabilities import validate_pi_kwargs

    validate_pi_kwargs(native_options)
    metadata = json.loads(json.dumps(model["metadata"], allow_nan=False))
    metadata.setdefault("cost_status", "unknown")
    if not native_output(condition):
        metadata["maxTokens"] = generation.get(
            "max_output_tokens", settings["budgets"].get("max_output_tokens", 4096)
        )
    else:
        metadata["maxTokens"] = min(metadata["maxTokens"], metadata["contextWindow"])
    api = (
        "openai-completions"
        if model["adapter"] == "openai-completions"
        else "openai-responses"
    )
    identifier = model.get("model_id") or model["id"]
    metadata["id"] = identifier
    return {
        "import_path": "simready_bench.harnesses:BenchmarkPi",
        "model_name": "openai/" + identifier,
        "skills": [str(Path(p).absolute()) for p in skill_dirs],
        "mcp_servers": [
            {
                "name": "benchmark",
                "transport": "stdio",
                "command": "python3",
                "args": ["-m", "simready_bench.engine.actor_mcp"],
            }
        ],
        "override_timeout_sec": settings["budgets"]["agent_seconds"],
        "env": {
            "OPENAI_API_KEY": DEFERRED_TOKEN,
            "SIMREADY_EPISODE_TOKEN": DEFERRED_TOKEN,
            "OPENAI_BASE_URL": "http://127.0.0.1:32190/v1",
        },
        "kwargs": {
            **(
                {"retry_owner": native_options["retry_owner"]}
                if "retry_owner" in native_options
                else {}
            ),
            "model_metadata": metadata,
            **({"continue_on_length": True} if condition.get('authoring_protocol', '').endswith('-continue-length-v1') else {}),
            "native_bridge_id": bridge_id,
            "version": "0.85.1",
            "native_output_policy": NATIVE_OUTPUT_POLICY
            if native_output(condition)
            else "bounded",
            "model_api": api,
            "system_prompt": prompt,
            "thinking": generation.get("reasoning_effort")
            or ("medium" if metadata.get("reasoning") else "off"),
            "max_turns": settings["budgets"]["inference_requests"],
        },
    }


def native_services_class():
    """Read-only skill mounts and actor isolation checks."""
    from simready_bench.engine.harbor_services import ReferenceNativeServices

    class SkillServices(ReferenceNativeServices):
        def __init__(
            self,
            *args,
            skill_root=None,
            budgets=None,
            audit_dir=None,
            reference_root=None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.skill_root = Path(skill_root).absolute() if skill_root else None
            self.check_dir = Path(audit_dir) if audit_dir else None
            self.reference_root = (
                Path(reference_root).absolute() if reference_root else None
            )
            self.container_checks = {}

        def actor_mounts_for_directory(self, environment_dir, actor_root):
            mounts = super().actor_mounts_for_directory(environment_dir, actor_root)
            if self.skill_root is not None:
                root = self.skill_root
                if (
                    root.resolve() != root
                    or root.stat().st_mode & 146
                    or any(
                        (
                            p.is_symlink() or p.stat().st_mode & 146
                            for p in root.rglob("*")
                        )
                    )
                ):
                    raise ValueError("Exact read-only public skill closure required")
                mounts.append(
                    {
                        "type": "bind",
                        "source": str(root),
                        "target": "/simready-bench/skills",
                        "read_only": True,
                        "bind": {"create_host_path": False},
                    }
                )
            return mounts

        async def record_container_audit(
            self, environment, identifier, role, expected_source=None
        ):
            if self.check_dir is None:
                return
            import re

            if not re.fullmatch("[a-f0-9]{64}", identifier):
                raise ValueError("Actual owned container identity required")
            template = '{"Mounts":{{json .Mounts}},"User":{{json .Config.User}},"NetworkMode":{{json .HostConfig.NetworkMode}},"DeviceRequests":{{json .HostConfig.DeviceRequests}}}'
            raw = json.loads(
                await environment._engine("inspect", "--format", template, identifier)
            )
            mounts = [
                {k: m[k] for k in ("Type", "Source", "Destination", "RW")}
                for m in raw["Mounts"]
            ]
            if expected_source is not None and (
                not any(
                    (
                        m["Destination"] == "/run/simready-verifier"
                        and m["Source"] == expected_source
                        for m in mounts
                    )
                )
            ):
                return
            if raw["NetworkMode"] != "none" or raw["DeviceRequests"]:
                raise ValueError(
                    "Native actor/verifier must have network none and no GPU requests"
                )
            if role == "actor":
                for destination in ("/task", "/run/simready-services"):
                    if not any(
                        (
                            m["Destination"] == destination and m["RW"] is False
                            for m in mounts
                        )
                    ):
                        raise ValueError(
                            "Actual public/service mount must be read-only"
                        )
                if self.skill_root is not None and (
                    not any(
                        (
                            m["Destination"] == "/simready-bench/skills"
                            and m["RW"] is False
                            for m in mounts
                        )
                    )
                ):
                    raise ValueError("Actual skill mount must be read-only")
                if any(
                    (
                        m["Destination"].startswith("/run/simready-verifier")
                        or (
                            self.reference_root is not None
                            and Path(m["Source"]).is_relative_to(self.reference_root)
                        )
                        for m in mounts
                    )
                ):
                    raise ValueError(
                        "Private verifier/reference mount forbidden in actor"
                    )
            receipt = {
                "schema": "simready-bench-container-audit-v1",
                "role": role,
                "container_id": identifier,
                "mounts": mounts,
                "user": raw["User"],
                "network_mode": raw["NetworkMode"],
                "device_requests": raw["DeviceRequests"],
                "environment_values_recorded": False,
            }
            self.check_dir.mkdir(parents=True, exist_ok=True, mode=448)
            target = self.check_dir / (role + ".json")
            if target.exists():
                raise ValueError("Container audit already frozen")
            target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
            target.chmod(384)
            self.container_checks[role] = {
                "path": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        async def bind_episode(self, event, env):
            await super().bind_episode(event, env)
            proof = env.network_isolation_proof
            if self.check_dir is not None:
                if not proof:
                    raise RuntimeError("Actual actor isolation receipt required")
                await self.record_container_audit(env, proof["container_id"], "actor")
            if (
                event.config.agent.import_path
                == "simready_bench.harbor_agents:ScopedNativeAgent"
            ):
                from simready_bench.harbor_agents import is_scoped_native_agent

                agents = [
                    a
                    for a in self.lifecycle._deferred_agents
                    if str(a.context_id) == str(event.trial_id)
                ]
                if len(agents) != 1 or not is_scoped_native_agent(agents[0]):
                    raise RuntimeError(
                        "Exactly one installed scoped native agent must bind to actual trial"
                    )
            if event.config.agent.import_path == "simready_bench.harnesses:BenchmarkPi":
                agents = [
                    a
                    for a in self.lifecycle._deferred_agents
                    if str(a.context_id) == str(event.trial_id)
                ]
                if len(agents) != 1 or type(agents[0]) is not BenchmarkPi:
                    raise RuntimeError(
                        "Exactly one Pi agent must bind to native episode"
                    )

    return SkillServices
