"""Narrow image/configuration glue for Harbor's pinned, unmodified Pi loop."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
from uuid import UUID
from pathlib import Path
import shlex
from typing import Any

from harbor.agents.installed.pi import Pi, _PI_MCP_ADAPTER_PACKAGE
from harbor.agents.model_connection import ResolvedModelConnection
from simready_bench.engine.native_output import NATIVE_OUTPUT_POLICY, EXTENSION_PATH

PI_VERSION = "0.85.1"
MCP_BRIDGE_PACKAGE = "pi-mcp-adapter@2.34.0"
THUMBNAIL = "@/task/thumbnail.png"
DEFERRED_TOKEN = "__SIMREADY_DEFERRED_EPISODE_TOKEN__"
_NATIVE_ENDPOINT = "http://127.0.0.1:32190"
_MODEL_FIELDS = {
    "id",
    "name",
    "input",
    "contextWindow",
    "maxTokens",
    "reasoning",
    "cost",
    "compat",
    "thinkingLevelMap",
}
_COST_FIELDS = {"input", "output", "cacheRead", "cacheWrite"}


def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            "model_metadata must explicitly describe the configured vision model"
        )
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("model_metadata must contain finite JSON values") from exc
    for required in ("input", "contextWindow", "maxTokens", "reasoning"):
        if required not in result:
            raise ValueError(f"model_metadata is missing {required}")
    inputs = result["input"]
    if (
        not isinstance(inputs, list)
        or set(inputs) != {"text", "image"}
        or len(inputs) != 2
    ):
        raise ValueError("model_metadata.input must explicitly include text and image")
    for key in ("contextWindow", "maxTokens"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"model_metadata.{key} must be a positive integer")
    if result["maxTokens"] > result["contextWindow"]:
        raise ValueError("model_metadata.maxTokens exceeds contextWindow")
    if type(result["reasoning"]) is not bool:
        raise ValueError("model_metadata.reasoning must be explicit boolean")
    if "cost" in result:
        cost = result["cost"]
        if not isinstance(cost, dict) or set(cost) != _COST_FIELDS:
            raise ValueError(
                "model_metadata.cost requires input/output/cacheRead/cacheWrite"
            )
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in cost.values()
        ):
            raise ValueError(
                "model_metadata.cost must contain finite nonnegative prices"
            )
        if result.get("cost_status") not in (None, "known"):
            raise ValueError(
                "model_metadata.cost_status conflicts with explicit prices"
            )
        result["cost_status"] = "known"
    elif result.get("cost_status") != "unknown":
        raise ValueError(
            "model_metadata requires real prices or explicit cost_status=unknown"
        )
    unknown = set(result) - _MODEL_FIELDS - {"cost_status"}
    if unknown:
        raise ValueError(
            "Unsupported model_metadata fields: " + ", ".join(sorted(unknown))
        )
    for key in ("compat", "thinkingLevelMap"):
        if key in result and not isinstance(result[key], dict):
            raise ValueError(f"model_metadata.{key} must be a JSON object")
    for key in ("id", "name"):
        if key in result and (
            not isinstance(result[key], str) or not result[key].strip()
        ):
            raise ValueError(f"model_metadata.{key} must be a nonempty string")
    return result


def _native_options(bridge_id, env):
    if bridge_id is None:
        return
    try:
        if not isinstance(bridge_id, str) or str(UUID(bridge_id)) != bridge_id:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(
            "native_bridge_id requires a canonical UUID for deferred binding"
        ) from error
    env = env or {}
    if any(
        env.get(name) != DEFERRED_TOKEN
        for name in ("OPENAI_API_KEY", "SIMREADY_EPISODE_TOKEN")
    ):
        raise ValueError(
            "Native deferred credentials require explicit noncredential markers"
        )
    if env.get("OPENAI_BASE_URL") != _NATIVE_ENDPOINT + "/v1":
        raise ValueError(
            "Native deferred inference requires the fixed loopback relay endpoint"
        )
    for name in env:
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD|AUTH", name, re.I) and name not in (
            "OPENAI_API_KEY",
            "SIMREADY_EPISODE_TOKEN",
        ):
            raise ValueError(
                "Native deferred configuration contains an unsupported credential field"
            )


def _native_output_metadata(bridge_id, metadata, output_policy="bounded"):
    if output_policy not in ("bounded", NATIVE_OUTPUT_POLICY):
        raise ValueError("Unsupported native output policy")
    if output_policy == NATIVE_OUTPUT_POLICY:
        if bridge_id is None:
            raise ValueError("Native output policy requires trusted episode binding")
        if metadata.get("compat", {}).get("supportsMaxOutputTokens") is not False:
            raise ValueError(
                "Native output policy must omit the Responses output field"
            )
        return
    if bridge_id is None:
        return
    # The actual pinned Responses builder derives max_output_tokens from this
    # configured model ceiling. It clamps values below 16 and can omit the field
    # via compat. Reject those cases before any actor setup or inference.
    if not 16 <= metadata["maxTokens"] <= metadata["contextWindow"]:
        raise ValueError("Native Pi configured output ceiling must be at least 16 and within its context window")
    if metadata.get("compat", {}).get("supportsMaxOutputTokens", True) is not True:
        raise ValueError("Native Pi must preserve explicit max_output_tokens")


def _compatibility(version: str | None) -> None:
    if version != PI_VERSION:
        raise ValueError(f"PiAgent requires Pi {PI_VERSION}")
    if (
        importlib.metadata.version("harbor") != "0.24.0"
        or _PI_MCP_ADAPTER_PACKAGE != MCP_BRIDGE_PACKAGE
    ):
        raise ValueError(
            "PiAgent requires the pinned Harbor 0.24.0/Pi MCP 2.34.0 compatibility set"
        )


class PiAgent(Pi):
    """Use Harbor's Pi agent with explicit real model metadata and image input.

    cost_status=unknown is retained as experiment metadata and is never converted
    to a claim of zero inference cost. The trusted gateway owns cost accounting;
    Pi's default zero-price telemetry is not authoritative.
    """

    def __init__(
        self,
        *args,
        model_metadata=None,
        version=PI_VERSION,
        native_bridge_id=None,
        native_output_policy="bounded",
        **kwargs,
    ):
        _compatibility(version)
        _native_options(native_bridge_id, kwargs.get("extra_env"))
        if native_bridge_id is not None:
            model = kwargs.get("model_name", "")
            if not isinstance(model, str) or model.split("/", 1)[0] not in (
                "openai",
                "openai-responses",
            ):
                raise ValueError(
                    "Native Pi provider must use the supported scoped OpenAI gateway"
                )
        self.native_bridge_id = native_bridge_id
        self._native_bound_episode = None
        # Consume this project-only keyword before Harbor's strict PiOptions.
        self.model_metadata = _metadata(model_metadata)
        self.native_output_policy = native_output_policy
        _native_output_metadata(
            native_bridge_id, self.model_metadata, self.native_output_policy
        )
        super().__init__(*args, version=version, **kwargs)
        self._check_reasoning()
        if native_bridge_id is not None:
            from simready_bench.engine.harbor import register_deferred_agent

            register_deferred_agent(native_bridge_id, self)

    def _env_sources(self):
        sources = super()._env_sources()
        # The native condition has an explicit per-episode connection. Never
        # discover provider credentials or transport settings in host env.
        return (
            tuple(source for source in sources if source is not os.environ)
            if self.native_bridge_id
            else sources
        )

    def bind_native_episode(self, bridge_id, episode_id, token, tool_url):
        """Bind only this native trial's in-memory scoped token; never process env."""
        if (
            self.native_bridge_id is None
            or bridge_id != self.native_bridge_id
            or self.context_id is None
            or str(self.context_id) != str(episode_id)
        ):
            raise RuntimeError(
                "Deferred native binding does not match the actual Harbor trial"
            )
        if (
            not isinstance(token, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{32,512}", token)
            or token == DEFERRED_TOKEN
        ):
            raise ValueError("A scoped native episode token is required")
        if tool_url.rstrip("/") != _NATIVE_ENDPOINT:
            raise ValueError("Native tools require the fixed loopback relay endpoint")
        if self._native_bound_episode is not None:
            if (
                self._native_bound_episode == str(episode_id)
                and self._extra_env["OPENAI_API_KEY"] == token
            ):
                return
            raise RuntimeError("Native trial credentials cannot be rebound")
        self._extra_env.update(
            OPENAI_API_KEY=token, SIMREADY_EPISODE_TOKEN=token, PI_OFFLINE="1"
        )
        self._native_bound_episode = str(episode_id)

    def _require_native_binding(self):
        if self.native_bridge_id is not None and self._native_bound_episode is None:
            raise RuntimeError(
                "Native Pi inference remains unavailable until the actual episode is bound"
            )

    async def run(self, instruction, environment, context):
        """Keep stock Pi execution/accounting; accept confirmed submission and configured limits."""
        from simready_bench.engine.harbor import NativeActorStop

        try:
            return await super().run(instruction, environment, context)
        except NativeActorStop as signal:
            bridge = getattr(environment, "bridge", None)
            if bridge is None or not bridge.accept_actor_stop(
                self, environment, signal
            ):
                raise
            from dataclasses import asdict

            self.actor_stop_receipt = asdict(signal.receipt)
            # No fabricated tool result or final assistant message: Harbor synchronizes
            # the real, interrupted-at-submit session using its ordinary finally path.
            return None

    @staticmethod
    def _build_max_turns_extension(max_turns: int) -> str:
        """Show each round's remaining calls using the same counter as Pi's stop."""
        return f"""export default function (pi) {{
  const maxTurns = {max_turns};
  let completedTurns = 0;
  const noticeType = "simready-authoring-round";

  pi.on("before_agent_start", (event) => ({{
    systemPrompt: `${{event.systemPrompt}}\n\nYou have a hard budget of ${{maxTurns}} model turns. Complete the task and provide your final answer within that budget.`,
  }}));

  // Pi 0.85.1 supplies a transient context copy before every model call.
  pi.on("context", (event) => {{
    const round = Math.min(completedTurns + 1, maxTurns);
    const remaining = maxTurns - round;
    let content = `Authoring round ${{round}} of ${{maxTurns}}. ${{remaining}} model round${{remaining === 1 ? "" : "s"}} remain${{remaining === 1 ? "s" : ""}} after this round.`;
    if (remaining === 0) {{
      content += " This is your final model round. Save the USD asset and required files, then submit now. No further model calls will be available.";
    }}
    const messages = event.messages.filter(
      (message) => message.role !== "custom" || message.customType !== noticeType,
    );
    return {{ messages: [...messages, {{
      role: "custom", customType: noticeType, content,
      display: false, timestamp: Date.now(),
    }}] }};
  }});

  pi.on("turn_end", (_event, ctx) => {{
    completedTurns += 1;
    if (completedTurns >= maxTurns) {{
      ctx.abort();
    }}
  }});
}}
"""

    async def install(self, environment) -> None:
        """Verify the prebuilt closure, without the stock network installer."""
        import simready_bench.engine.pi_runtime as frozen_pi

        source = Path(frozen_pi.__file__).read_text()
        result = await self.exec_as_agent(
            environment,
            command=". ~/.nvm/nvm.sh && python3 -c " + shlex.quote(source),
            timeout_sec=90,
        )
        try:
            receipt = json.loads(result.stdout)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "Frozen Pi setup produced no integrity receipt"
            ) from error
        if (
            receipt.get("status") != "frozen_pi_runtime_verified"
            or receipt.get("uid") != 1000
        ):
            raise RuntimeError("Frozen Pi setup integrity verification failed")
        destination = self.logs_dir / "setup"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "frozen-runtime.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )

    def _build_mcp_config(self):
        config = super()._build_mcp_config()
        if self.native_bridge_id is not None:
            # Preview service deadline is90s; leave transport/IPC headroom.
            # This is the actual pinned bridge's supported configuration field.
            config["settings"]["requestTimeoutMs"] = 120000
        return config

    def _check_reasoning(self):
        if (
            self.options.thinking not in (None, "off")
            and not self.model_metadata["reasoning"]
        ):
            raise ValueError(
                "Configured thinking requires model_metadata.reasoning=true"
            )

    @classmethod
    def preflight(cls, kwargs=None, env=None):
        # Native Harbor calls preflight before queueing. Keep upstream validation
        # after removing only this subclass's extra configuration field.
        options = dict(kwargs or {})
        metadata = _metadata(options.pop("model_metadata", None))
        bridge_id = options.pop("native_bridge_id", None)
        output_policy = options.pop("native_output_policy", "bounded")
        _native_options(bridge_id, env)
        _native_output_metadata(bridge_id, metadata, output_policy)
        _compatibility(options.get("version", PI_VERSION))
        super().preflight(options, env)
        parsed = cls.parse_options(options, env)
        if parsed.thinking not in (None, "off") and not metadata["reasoning"]:
            raise ValueError(
                "Configured thinking requires model_metadata.reasoning=true"
            )

    def _build_custom_models_json(
        self,
        access: ResolvedModelConnection,
        model_id: str,
    ) -> dict[str, Any]:
        self._require_native_binding()
        # Preserve upstream provider/API/key-reference and reasoning behavior.
        models_json = super()._build_custom_models_json(access, model_id)
        if models_json is None:
            raise ValueError(
                "PiAgent requires an explicitly configured inference gateway endpoint"
            )
        # Revalidate at the actor-config boundary as well as construction.
        # Transport headers/API/endpoints belong only to the upstream builder.
        metadata = _metadata(self.model_metadata)
        _native_output_metadata(
            self.native_bridge_id, metadata, self.native_output_policy
        )
        if metadata.get("id", model_id) != model_id:
            raise ValueError("model_metadata.id does not match the selected model")
        for provider in models_json["providers"].values():
            for model in provider["models"]:
                if model["id"] != model_id:
                    continue
                # Experiment-only pricing status is not a Pi models.json field.
                model.update({k: v for k, v in metadata.items() if k in _MODEL_FIELDS})
                model["id"] = model_id
        return models_json

    def build_cli_flags(self) -> str:
        self._require_native_binding()
        flags = super().build_cli_flags()
        if self.native_output_policy == NATIVE_OUTPUT_POLICY:
            flags += " --extension " + shlex.quote(EXTENSION_PATH)
        return " ".join(part for part in (flags, shlex.quote(THUMBNAIL)) if part)
