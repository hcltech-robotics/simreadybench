"""Thin extension of the installed Harbor Pi adapter, not a new agent loop."""

import importlib.metadata
from pathlib import Path

from pydantic import Field
from harbor.agents.installed.pi import Pi, PiOptions

from . import VERSION
from .profile import ENDPOINT, PI_VERSION, ModelProfile

ROOT = Path(__file__).parent


class NvidiaPiOptions(PiOptions):
    model_profile: ModelProfile
    require_images: bool = Field(default=True)


class NvidiaPi(Pi):
    options_model = NvidiaPiOptions

    @staticmethod
    def name():
        return "pi-nvidia-feedback"

    def __init__(self, *args, **kwargs):
        if importlib.metadata.version("harbor") != "0.24.0":
            raise RuntimeError("This adapter targets Harbor 0.24.0; reverify other builds")
        kwargs.setdefault("version", PI_VERSION)
        kwargs.setdefault("model_api", "openai-completions")
        super().__init__(*args, **kwargs)
        profile = self.options.model_profile
        if self._version != PI_VERSION or self.options.model_api != "openai-completions":
            raise ValueError("Pi version and NVIDIA Chat Completions API are pinned")
        if self.skills_dir or self.mcp_servers:
            raise ValueError("This plain Pi variant does not add skills or MCP tools")
        if self.model_name != "nvidia/" + profile.id:
            raise ValueError("model_name must match nvidia/ plus the exact profile ID")
        if not profile.supports(self.options.thinking):
            raise ValueError("Select an explicitly supported thinking level; no silent clamping")
        if self.options.require_images and "image" not in profile.input:
            raise ValueError("This image task requires a model with verified image input")
        self._usage_missing = False

    def _build_custom_models_json(self, access, model_id):
        if access.configured_base_url != ENDPOINT:
            raise ValueError("Set NVIDIA_BASE_URL to the documented NVIDIA Build endpoint")
        # Parent resolves only the environment-variable name, never serializes a key.
        config = super()._build_custom_models_json(access, model_id)
        provider = config["providers"]["harbor-endpoint"]
        provider["models"] = [self.options.model_profile.pi_model()]
        provider["authHeader"] = True
        return config

    def render_instruction(self, instruction):
        return (ROOT / "feedback.txt").read_text() + "\n\n<benchmark-instruction>\n" + super().render_instruction(instruction) + "\n</benchmark-instruction>"

    @classmethod
    def _metrics_from_usage(cls, usage):
        required = ("input", "output", "cacheRead", "cacheWrite")
        if (not isinstance(usage, dict)
                or any(type(usage.get(k)) is not int or usage[k] < 0 for k in required)
                or sum(usage[k] for k in required) == 0):
            # Pi's provider initializes all counters to zero before streamed usage
            # arrives. That shape alone is not evidence of a known zero-token call.
            return None
        metrics = super()._metrics_from_usage(usage)
        if metrics:
            metrics.extra = {**(metrics.extra or {}), "pricing": "unknown",
                             "pi_reported_cost": usage.get("cost")}
            metrics.cost_usd = None  # Pi defaults missing model prices to zero.
        return metrics

    def _accounting_metrics(self, events, copied_until):
        self._usage_missing = False
        for position, event in events:
            if position <= copied_until:
                continue
            message = event.get("message") or {}
            if event.get("type") in {"compaction", "branch_summary"}:
                usage = event.get("usage")
            elif event.get("type") == "message" and message.get("role") == "assistant":
                usage = message.get("usage")
            else:
                continue
            if self._metrics_from_usage(usage) is None:
                self._usage_missing = True
        return super()._accounting_metrics(events, copied_until)

    def _convert_session_events_to_trajectory(self, events, copied_until=0):
        self._accounting_metrics(events, copied_until)
        trajectory = super()._convert_session_events_to_trajectory(events, copied_until)
        if trajectory:
            trajectory.agent.name = self.name()
            trajectory.agent.extra = {**(trajectory.agent.extra or {}), "variant_version": VERSION,
                "pi_version": PI_VERSION, "model_profile": self.options.model_profile.model_dump(),
                "requested_thinking": self.options.thinking,
                "pricing": "unknown", "usage_complete": not self._usage_missing}
            trajectory.final_metrics.total_cost_usd = None
            if self._usage_missing:
                trajectory.final_metrics.total_prompt_tokens = None
                trajectory.final_metrics.total_completion_tokens = None
                trajectory.final_metrics.total_cached_tokens = None
        return trajectory

    def _populate_context_from_output(self, context):
        # A stdout fallback cannot prove all calls were retained. Keep it unknown.
        context.n_input_tokens = context.n_output_tokens = context.n_cache_tokens = None
        context.cost_usd = None
        self._usage_missing = True

    def populate_context_post_run(self, context):
        super().populate_context_post_run(context)
        context.cost_usd = None
        if self._usage_missing:
            context.n_input_tokens = context.n_output_tokens = context.n_cache_tokens = None
        context.metadata = {**(context.metadata or {}), "pi_nvidia": {
            "version": VERSION, "pi_version": PI_VERSION, "pricing": "unknown",
            "requested_thinking": self.options.thinking,
            "usage_complete": not self._usage_missing,
            "model_profile": self.options.model_profile.model_dump()}}
