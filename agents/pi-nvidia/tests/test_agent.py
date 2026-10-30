import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.model_connection import ResolvedModelConnection
from harbor.models.agent.context import AgentContext

from pi_nvidia.agent import NvidiaPi
from pi_nvidia.profile import ModelProfile, THINKING


def profile(**changes):
    # Synthetic capabilities for offline fixtures, not an NVIDIA model selection.
    return {"id": "fixture/vision", "input": ["text", "image"], "contextWindow": 32000,
            "maxTokens": 4000, "reasoning": False, "toolCalls": True, "thinkingLevelMap": {},
            "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                       "supportsReasoningEffort": False, "supportsUsageInStreaming": True,
                       "maxTokensField": "max_tokens"},
            "evidence": "synthetic offline test fixture", **changes}


def agent(tmp_path, data=None, **kwargs):
    data = data or profile()
    return NvidiaPi(logs_dir=tmp_path, model_name="nvidia/" + data["id"],
                    model_profile=data, thinking=kwargs.pop("thinking", "off"), **kwargs)


def connection():
    # A mock value exercises upstream env-name resolution; never sent to a backend.
    return ResolvedModelConnection(provider="nvidia", api_key="offline-fixture",
        configured_base_url="https://integrate.api.nvidia.com/v1",
        env={"NVIDIA_API_KEY": "offline-fixture"})


def test_capabilities_and_secret_reference_are_explicit(tmp_path):
    a = agent(tmp_path)
    config = a._build_custom_models_json(connection(), "fixture/vision")
    p = config["providers"]["harbor-endpoint"]
    assert p["api"] == "openai-completions" and p["apiKey"] == "$NVIDIA_API_KEY"
    assert p["authHeader"] is True
    model = p["models"][0]
    assert model["input"] == ["text", "image"]
    assert (model["contextWindow"], model["maxTokens"]) == (32000, 4000)
    assert model["compat"]["maxTokensField"] == "max_tokens"
    assert "cost" not in model and "offline-fixture" not in json.dumps(config)
    assert a._version == "1.4.0" and not a.skills_dir and not a.mcp_servers


def test_native_harbor_factory_selects_the_adapter_without_builtin_name(tmp_path):
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig
    config = AgentConfig(import_path="pi_nvidia.agent:NvidiaPi",
        model_name="nvidia/fixture/vision", kwargs={"model_profile": profile(), "thinking": "off"})
    a = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(a, NvidiaPi) and a._version == "1.4.0"


@pytest.mark.parametrize("change", [{"contextWindow": None}, {"maxTokens": 64000},
    {"compat": {}}, {"id": "SELECT_MODEL"}, {"apiKey": "forbidden"}])
def test_unspecified_or_inconsistent_capabilities_fail(change):
    with pytest.raises(ValueError):
        ModelProfile.model_validate(profile(**change))


def test_thinking_is_model_specific_and_never_silently_clamped(tmp_path):
    mapping = {level: None for level in THINKING}
    mapping.update(off="off", high="large")
    data = profile(reasoning=True, thinkingLevelMap=mapping)
    data["compat"]["thinkingFormat"] = "openai"
    data["compat"]["supportsReasoningEffort"] = True
    a = agent(tmp_path / "a", data, thinking="high")
    m = a._build_custom_models_json(connection(), data["id"])["providers"]["harbor-endpoint"]["models"][0]
    assert m["thinkingLevelMap"]["high"] == "large" and m["thinkingLevelMap"]["low"] is None
    with pytest.raises(ValueError, match="no silent clamping"):
        agent(tmp_path / "b", data, thinking="low")
    with pytest.raises(ValueError, match="no silent clamping"):
        agent(tmp_path / "c", data, thinking=None)
    b = agent(tmp_path / "d", profile(id="fixture/other"))
    assert b.options.thinking == "off" and a.options.thinking == "high"


def test_text_only_model_is_not_misrepresented_as_vision(tmp_path):
    data = profile(input=["text"])
    with pytest.raises(ValueError, match="image task"):
        agent(tmp_path, data)
    a = agent(tmp_path, data, require_images=False)
    assert a.options.model_profile.input == ["text"]


def test_prompt_policy_preserves_original_and_allows_early_completion(tmp_path):
    original = "Original frozen observation: /task/observation.png. Original task unchanged."
    prompt = agent(tmp_path).render_instruction(original)
    assert original in prompt
    for text in ("http://feedback:8080/health", "validate_candidate", "limit, used and remaining",
                 "Reserve the last available call", "same request ID", "Retry-After",
                 "Unknown checks", "five model inference calls", "passing first candidate"):
        assert text.lower() in prompt.lower()
    assert "not the feedback client" in prompt


def test_existing_pi_run_rewrites_profile_each_trial_without_install(tmp_path, monkeypatch):
    configs, commands = [], []
    a = agent(tmp_path)
    monkeypatch.setattr(type(a), "model_connection", property(lambda self: connection()))
    async def write(env, config):
        configs.append(copy.deepcopy(config))
    async def execute(environment, command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(stdout="", return_code=0)
    monkeypatch.setattr(a, "_write_custom_models_json", write)
    monkeypatch.setattr(a, "exec_as_agent", execute)
    for _ in range(2):
        asyncio.run(a.run("ORIGINAL", object(), AgentContext()))
    assert len(configs) == 2
    cmd = commands[-1][0]
    assert "--provider harbor-endpoint --model fixture/vision" in cmd
    assert "--thinking off" in cmd and "ORIGINAL" in cmd and "validate_candidate" in cmd
    assert "--mode json" in cmd and "cadgen" not in cmd and "build123d" not in cmd


def events(usage=True):
    rows = [{"type": "session", "id": "fixture-session", "timestamp": "2026-10-30T00:00:00Z"}]
    for n in range(2):
        message = {"role": "assistant", "content": [{"type": "text", "text": "done"}],
                   "api": "openai-completions", "provider": "harbor-endpoint",
                   "model": "fixture/vision", "stopReason": "stop", "timestamp": 1793318400000}
        if usage or n == 0:
            message["usage"] = {"input": 100, "output": 10, "cacheRead": 80,
                                "cacheWrite": 0, "totalTokens": 190,
                                "cost": {"total": 123.45}}
        rows.append({"type": "message", "message": message})
    return list(enumerate(rows, start=1))


def test_tokens_are_kept_but_unverified_prices_are_not(tmp_path):
    a = agent(tmp_path)
    metrics = a._accounting_metrics(events(), 0)
    assert sum(m.prompt_tokens for m in metrics) == 360
    assert sum(m.cached_tokens for m in metrics) == 160
    assert all(m.cost_usd is None for m in metrics)
    # Copied prefix of a resumed session must not be charged again.
    assert len(a._accounting_metrics(events(), 2)) == 1
    trajectory = a._convert_session_events_to_trajectory(events())
    assert trajectory.final_metrics.total_cost_usd is None
    assert trajectory.final_metrics.total_prompt_tokens == 360


def test_missing_usage_leaves_totals_unknown_and_cost_unknown(tmp_path):
    a = agent(tmp_path)
    trajectory = a._convert_session_events_to_trajectory(events(usage=False))
    assert trajectory.final_metrics.total_prompt_tokens is None
    assert trajectory.agent.extra["usage_complete"] is False
    assert trajectory.final_metrics.total_cost_usd is None
    context = AgentContext()
    a._populate_context_from_output(context)
    assert context.n_input_tokens is None and context.cost_usd is None


def test_pi_default_zero_usage_is_not_proof_of_free_or_zero_token_inference(tmp_path):
    a = agent(tmp_path)
    assert a._metrics_from_usage({"input": 0, "output": 0, "cacheRead": 0,
                                  "cacheWrite": 0, "cost": {"total": 0}}) is None


def test_parent_image_content_conversion_is_preserved(tmp_path):
    from itertools import count
    a = agent(tmp_path)
    # Tiny base64 payload tests transport preservation, not decoding/vision quality.
    content = a._content([{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}],
                         next_image_index=count(1))
    assert isinstance(content, list) and content[0].type == "image"
    assert (tmp_path / "images/pi_image_1.png").read_bytes() == b"hello"


@pytest.mark.parametrize("complete", [True, False])
def test_actual_harbor_context_population_from_session_receipts(tmp_path, complete):
    a = agent(tmp_path)
    sessions = tmp_path / "pi/sessions"
    sessions.mkdir(parents=True)
    (sessions / "fixture.jsonl").write_text("".join(json.dumps(e) + "\n" for _, e in events(usage=complete)))
    context = AgentContext()
    a.populate_context_post_run(context)
    assert context.cost_usd is None
    assert context.metadata["pi_nvidia"]["usage_complete"] is complete
    assert context.n_input_tokens == (360 if complete else None)
    assert context.n_cache_tokens == (160 if complete else None)
    assert context.n_output_tokens == (20 if complete else None)
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert trajectory["final_metrics"].get("total_cost_usd") is None
