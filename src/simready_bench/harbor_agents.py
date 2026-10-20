"""Scoped connection glue around Harbor's actual installed agent classes.

AgentFactory owns the agent registry and each agent's run loop. This module only
binds an opaque episode credential, the existing relay and preinstalled binaries.
"""

from __future__ import annotations
import copy, importlib.metadata, json, os, re
from uuid import UUID
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.agent.name import AgentName
from simready_bench.engine.pi import DEFERRED_TOKEN

ENDPOINT = "http://127.0.0.1:32190"
_WRAPPERS = {}


def resolve_native_class(*, agent=None, import_path=None):
    if importlib.metadata.version("harbor") != "0.24.0":
        raise ValueError("Pinned Harbor 0.24.0 required")
    if (agent is None) == (import_path is None):
        raise ValueError(
            "Select exactly one native agent name or registered import_path"
        )
    names = AgentFactory.registered_names()
    if agent is not None:
        if agent not in names:
            raise ValueError("Unknown installed native Harbor agent")
        cls = AgentFactory.get_agent_class(AgentName(agent))
        return (agent, cls)
    if not isinstance(import_path, str) or not re.fullmatch(
        "harbor\\.agents\\.[A-Za-z0-9_.]+:[A-Za-z0-9_]+", import_path
    ):
        raise ValueError("Exact installed Harbor registry import_path required")
    for name in names:
        try:
            cls = AgentFactory.get_agent_class(AgentName(name))
        except ImportError:
            continue
        if cls.__module__ + ":" + cls.__name__ == import_path:
            return (name, cls)
    raise ValueError("Native import_path is not in the installed Harbor registry")


def compatibility(name, cls):
    """Derive compatibility from native connection and image-tool declarations."""
    if name == "codex":
        return {"admitted": False, "reason": "project_policy_codex_is_builder_only"}
    if name == "pi":
        return {
            "admitted": True,
            "reason": "existing_native_" + name + "_integration",
            "integration": "pinned_vision_glue"
            if name == "pi"
            else "model_free_control",
        }
    if not issubclass(cls, BaseInstalledAgent):
        return {
            "admitted": False,
            "reason": "host_agent_loop_cannot_use_isolated_installed_cli_environment",
        }
    spec = getattr(cls, "MODEL_CONNECTION", None)
    if spec is None:
        return {
            "admitted": False,
            "reason": "native_agent_has_no_declared_model_connection_hook",
        }
    if spec.default_provider not in (None, "openai"):
        return {
            "admitted": False,
            "reason": "native_provider_protocol_"
            + str(spec.default_provider)
            + "_not_supported_by_scoped_openai_gateway",
        }
    if not (spec.passthrough or spec.base_url_envs):
        return {
            "admitted": False,
            "reason": "native_connection_does_not_expose_configurable_relay_endpoint",
        }
    if not getattr(cls.capabilities, "mcp_servers", False):
        return {
            "admitted": False,
            "reason": "native_agent_has_no_mcp_image_tool_delivery",
        }
    return {
        "admitted": True,
        "reason": "native_model_connection_and_mcp_hooks_available",
        "integration": "scoped_installed_native",
        "image_delivery": "benchmark_get_task_thumbnail_mcp",
        "setup_requirement": "explicit_version_pin_and_preinstalled_binary",
        "request_requirement": "all_native_requests_must_pass_same_gateway_payload_and_budget_guards",
    }


def validate_native_kwargs(name, cls, value):
    if value is None:
        value = {}
    if (
        type(value) is not dict
        or len(json.dumps(value, allow_nan=False).encode()) > 65536
    ):
        raise ValueError("Bounded JSON native kwargs object required")
    value = copy.deepcopy(value)
    fields = getattr(getattr(cls, "options_model", None), "model_fields", {})
    if set(value) - set(fields):
        raise ValueError(
            "Unknown options for installed native "
            + name
            + ": "
            + ", ".join(sorted(set(value) - set(fields)))
        )
    forbidden = {
        "config",
        "prompt_template_path",
        "api_key",
        "base_url",
        "api_base",
        "baseURL",
        "provider",
        "factory",
        "factory_path",
        "factory_kwargs",
        "install_command",
        "source",
        "repo_url",
        "repo_ref",
        "registry_entry",
        "registry_spec",
        "registry_ref",
        "registry_cache_dir",
        "registry_entry_path",
        "distribution_preference",
        "authenticate_method_id",
        "auth_policy",
    }

    def inspect(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in forbidden or re.search(
                    "(?:credential|secret|password|token_env|api_key|base_url|baseURL|endpoint|headers|command|(?:^|_)(?:path|file|dir)$)",
                    key,
                ):
                    raise ValueError(
                        "Native credential, endpoint, executable or controller-file option is managed by trusted integration: "
                        + key
                    )
                inspect(item)
        elif isinstance(node, list):
            for item in node:
                inspect(item)
        elif isinstance(node, str) and (
            "{file:" in node or "{env:" in node or "\x00" in node
        ):
            raise ValueError("Native host-file/environment interpolation is forbidden")

    inspect(value)
    if name != "pi":
        version = value.get("version")
        if (
            not isinstance(version, str)
            or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", version)
            or version in ("latest", "main", "master", "dev")
        ):
            raise ValueError(
                "Native harness requires an explicit preinstalled version pin"
            )
    cls.preflight(kwargs=value, env=deferred_environment(cls))
    return value


def deferred_environment(cls):
    spec = getattr(cls, "MODEL_CONNECTION", None)
    result = {
        "OPENAI_API_KEY": DEFERRED_TOKEN,
        "SIMREADY_EPISODE_TOKEN": DEFERRED_TOKEN,
        "OPENAI_BASE_URL": ENDPOINT + "/v1",
    }
    if spec is not None:
        for name in spec.api_key_envs:
            result[name] = DEFERRED_TOKEN
        for name in spec.base_url_envs:
            result[name] = ENDPOINT + "/v1"
    return result


def normalize_harness(entry):
    name, cls = resolve_native_class(
        agent=entry.get("agent"), import_path=entry.get("import_path")
    )
    support = compatibility(name, cls)
    if not support["admitted"]:
        raise ValueError("Native Harbor " + name + " unavailable: " + support["reason"])
    options = validate_native_kwargs(name, cls, entry.get("kwargs", {}))
    result = {**entry, "native_agent": name, "kwargs": options}
    if name == "pi":
        result["adapter"] = name
    return result


def _opencode_config(model, api_style, metadata, limits, system_prompt, generation):
    package = (
        "@ai-sdk/openai"
        if api_style == "openai-responses"
        else "@ai-sdk/openai-compatible"
    )
    config = {
        "enabled_providers": ["openai"],
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
        "compaction": {"auto": False, "prune": False},
        "provider": {
            "openai": {
                "npm": package,
                "options": {
                    "baseURL": ENDPOINT + "/v1",
                    "apiKey": "{env:OPENAI_API_KEY}",
                },
                "models": {
                    model: {
                        "name": model,
                        "limit": {
                            "context": min(
                                metadata["contextWindow"], limits["max_input_tokens"]
                            ),
                            "output": limits["max_output_tokens"],
                        },
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                    }
                },
            }
        },
    }
    if api_style == "openai-responses":
        config["provider"]["openai"]["models"][model]["options"] = {"store": False}
    if limits["max_output_tokens"] is None:
        config["provider"]["openai"]["models"][model]["limit"].pop("output")
    effort = generation.get("reasoning_effort")
    if effort not in (None, "off", "none"):
        config["provider"]["openai"]["models"][model].setdefault("options", {})[
            "reasoningEffort"
        ] = effort
    if system_prompt:
        config["agent"] = {"build": {"prompt": system_prompt}}
    return config


class _ScopedNativeMixin:
    def __init__(
        self,
        *args,
        native_bridge_id,
        native_api_style,
        native_model_metadata,
        native_limits,
        native_system_prompt="",
        native_generation=None,
        **kwargs,
    ):
        if (
            not isinstance(native_bridge_id, str)
            or str(UUID(native_bridge_id)) != native_bridge_id
        ):
            raise ValueError("Canonical installed native bridge ID required")
        expected = deferred_environment(type(self)._native_class)
        supplied = kwargs.get("extra_env", {})
        if supplied != expected:
            raise ValueError(
                "Only deferred scoped native connection markers are permitted"
            )
        self.native_bridge_id = native_bridge_id
        self._native_bound_episode = None
        self._native_system_prompt = native_system_prompt
        self._native_limits = copy.deepcopy(native_limits)
        self._native_api_style = native_api_style
        self._native_metadata = copy.deepcopy(native_model_metadata)
        if self._native_name == "opencode":
            owned = _opencode_config(
                kwargs["model_name"].split("/", 1)[1],
                native_api_style,
                native_model_metadata,
                native_limits,
                native_system_prompt,
                native_generation or {},
            )
            user = kwargs.get("opencode_config", {})
            kwargs["opencode_config"] = type(self)._native_class._deep_merge(
                copy.deepcopy(user), owned
            )
        super().__init__(*args, **kwargs)
        from simready_bench.engine.harbor import register_deferred_agent

        register_deferred_agent(native_bridge_id, self)

    def _env_sources(self):
        return tuple(
            (source for source in super()._env_sources() if source is not os.environ)
        )

    def bind_native_episode(self, bridge_id, episode_id, token, tool_url):
        if (
            bridge_id != self.native_bridge_id
            or str(self.context_id) != str(episode_id)
            or tool_url.rstrip("/") != ENDPOINT
        ):
            raise RuntimeError("Native agent binding differs from actual trial")
        if (
            not isinstance(token, str)
            or not re.fullmatch("[A-Za-z0-9_-]{32,512}", token)
            or token == DEFERRED_TOKEN
        ):
            raise ValueError("Opaque episode token required")
        if self._native_bound_episode is not None:
            if (
                self._native_bound_episode == str(episode_id)
                and self._extra_env["OPENAI_API_KEY"] == token
            ):
                return
            raise RuntimeError("Native episode token cannot be rebound")
        for key, value in deferred_environment(type(self)._native_class).items():
            self._extra_env[key] = token if value == DEFERRED_TOKEN else value
        self._native_bound_episode = str(episode_id)

    async def install(self, environment):
        command = super().get_version_command()
        if not command:
            raise RuntimeError(
                "Native binary version check unavailable; preinstall qualification required"
            )
        try:
            result = await self.exec_as_agent(
                environment, command=command, timeout_sec=10
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Preinstalled native "
                + self._native_name
                + " binary unavailable; build it into the actor image"
            ) from exc
        actual = self.parse_version(result.stdout or "").strip().removeprefix("v")
        expected = str(self._version).removeprefix("v")
        if actual != expected:
            raise RuntimeError(
                "Native " + self._native_name + " version differs from explicit pin"
            )

    def build_cli_flags(self):
        flags = super().build_cli_flags()
        if self._native_name == "opencode":
            flags = (flags + " --file /task/thumbnail.png").strip()
        return flags

    async def run(self, instruction, environment, context):
        if self._native_bound_episode is None:
            raise RuntimeError(
                "Native agent cannot infer before actual episode binding"
            )
        if self._native_system_prompt and self._native_name != "opencode":
            instruction = self._native_system_prompt + "\n\n" + instruction
        return await super().run(instruction, environment, context)


def scoped_native_class(name, cls):
    key = (name, cls)
    if key not in _WRAPPERS:
        _WRAPPERS[key] = type(
            "ScopedNative" + cls.__name__,
            (_ScopedNativeMixin, cls),
            {"__module__": __name__, "_native_class": cls, "_native_name": name},
        )
    return _WRAPPERS[key]


def is_scoped_native_agent(agent):
    return type(agent) in _WRAPPERS.values()


class ScopedNativeAgent(BaseInstalledAgent):
    """AgentFactory import-path constructor; returns the actual native subclass."""

    @staticmethod
    def name():
        return "simready-scoped-native"

    @classmethod
    def preflight(cls, kwargs=None, env=None):
        values = dict(kwargs or {})
        name, native = resolve_native_class(agent=values.get("native_agent"))
        support = compatibility(name, native)
        if not support["admitted"]:
            raise ValueError(support["reason"])
        validate_native_kwargs(name, native, values.get("native_kwargs", {}))
        if env != deferred_environment(native):
            raise ValueError("Deferred native credentials required")

    def __new__(cls, *args, native_agent, native_kwargs=None, **kwargs):
        name, native = resolve_native_class(agent=native_agent)
        support = compatibility(name, native)
        if not support["admitted"]:
            raise ValueError(support["reason"])
        options = validate_native_kwargs(name, native, native_kwargs)
        return scoped_native_class(name, native)(*args, **options, **kwargs)
