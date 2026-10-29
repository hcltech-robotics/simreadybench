import asyncio
import json
from pathlib import Path

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.trial.config import AgentConfig
from harbor.models.agent.acp_source import AcpSourceManifest

from cad_agent.acp import child_environment, transform_request
from cad_agent.native import CodexText2Cad
from cad_agent.workflow import bundle_identity, render_instruction, verify_bundle, verify_generation


def test_full_skill_loaded_and_original_instruction_preserved():
    pins = verify_bundle()
    assert len([p for p in pins["files"] if p.startswith("vendor/cad/")]) == 13
    original = "An unchanged task with <image> and a five-call feedback limit."
    rendered = render_instruction(original)
    assert original in rendered
    assert "<selected-cad-skill>" in rendered
    assert "export_asset" in rendered
    assert "millimeters" in rendered
    assert pins["skill_commit"] == "2781952aebca62cb4fd11a9d2f91b4e26ea48590"


@pytest.mark.parametrize("transport", ["native", "hosted"])
def test_validation_loop_reaches_agent_prompt_without_changing_task(tmp_path, transport):
    original = "Reconstruct the requested asset. Original task and five-call allowance."
    if transport == "native":
        agent = CodexText2Cad(logs_dir=tmp_path, model_name="openai/gpt-6.1-terra")
        prompt = agent.render_instruction(original)
        assert original in prompt
    else:
        blocks = [{"type": "text", "text": original},
                  {"type": "image", "data": "frozen-input", "mimeType": "image/png"}]
        request = {"method": "session/prompt", "params": {"sessionId": "s", "prompt": blocks}}
        result = transform_request(request, "/agent")
        assert result["params"]["prompt"][1:] == blocks
        assert request["params"]["prompt"] == blocks
        prompt = result["params"]["prompt"][0]["text"]
    # These are instructions delivered to the agent, not a claim of runtime enforcement.
    prompt = " ".join(prompt.split())
    assert "http://feedback:8080/health" in prompt
    assert "limit, used and remaining" in prompt
    assert "python -m simready_eval.feedback validate_candidate" in prompt
    assert "result.simready.all_scored_mandatory_pass" in prompt
    assert "Reserve the last available call" in prompt
    assert "fresh request ID" in prompt and "same request ID" in prompt
    assert "HTTP 200 or a zero client exit code" in prompt
    assert "USD, thumbnail and generation receipts" in prompt
    assert "no fixed authoring-turn limit" in prompt
    assert "hidden visual/kinematic fidelity scores" in prompt
    assert "calls used/remaining and unresolved findings" in prompt


def test_mcp_keeps_explicit_writable_cad_state(monkeypatch):
    from cad_agent.prepare import runtime_env
    monkeypatch.setenv("CADGEN_ENGINE_DIR", "/writable/engine")
    assert runtime_env("/agent")["CADGEN_ENGINE_DIR"] == "/writable/engine"
    assert runtime_env("/agent", "/new/state")["CADGEN_ENGINE_DIR"] == "/new/state/engine"


def test_native_factory_selects_variant_and_separates_model(tmp_path):
    config = AgentConfig(import_path="cad_agent.native:CodexText2Cad", model_name="openai/gpt-6.1-terra",
                         kwargs={"version": "0.164.0", "reasoning_effort": "medium"})
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
    assert isinstance(agent, CodexText2Cad)
    assert agent.name() == "codex-text2cad-build123d"
    assert agent.model_name == "openai/gpt-6.1-terra"
    assert agent.options.reasoning_effort == "medium"
    assert agent._base_config["mcp_servers"]["cad"]["required"]
    assert "selected-cad-skill" in agent.render_instruction("original task")


def test_variant_trajectory_and_parent_usage_are_preserved(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from harbor.agents.installed.codex import Codex
    from harbor.models.agent.context import AgentContext
    trajectory = SimpleNamespace(agent=SimpleNamespace(name="codex", version="0.164.0", extra={"native": "retained"}))
    monkeypatch.setattr(Codex, "convert_trajectory", lambda self, logs: trajectory)
    agent = CodexText2Cad(logs_dir=tmp_path, model_name="openai/gpt-6.1-terra")
    converted = agent.convert_trajectory(tmp_path)
    assert converted.agent.name == "codex-text2cad-build123d"
    assert converted.agent.version == "0.164.0" and converted.agent.extra["native"] == "retained"
    def populate(self, context):
        context.cost_usd = 0.012
        context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens = 7, 4, 1
    monkeypatch.setattr(Codex, "populate_context_post_run", populate)
    context = AgentContext(metadata={"original": True})
    agent.populate_context_post_run(context)
    assert (context.cost_usd, context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens) == (0.012, 7, 4, 1)
    assert context.metadata["original"] and context.metadata["cad_agent"]["id"] == agent.name()


def test_run_configures_baked_agent_before_parent_run(tmp_path, monkeypatch):
    """Install is deliberately never called: emulate Harbor's baked-agent path."""
    from harbor.agents.installed.codex import Codex
    from harbor.models.agent.context import AgentContext
    calls = []
    class Environment:
        async def download_dir(self, *args):
            calls.append(("download", args))
    class Agent(CodexText2Cad):
        async def exec_as_agent(self, environment, command, **kwargs):
            calls.append(("exec", command))
    async def parent_run(self, instruction, environment, context):
        calls.append(("parent", instruction))
    monkeypatch.setattr(Codex, "run", parent_run)
    agent = Agent(logs_dir=tmp_path, model_name="openai/gpt-6.1-sol", reasoning_effort="medium")
    asyncio.run(agent.run("original", Environment(), AgentContext()))
    assert "prepare trial" in calls[0][1]
    assert calls[1] == ("parent", "original")
    assert "prepare verify" in calls[2][1]


def test_acp_preserves_images_mcp_and_requests():
    message = {"jsonrpc": "2.0", "id": 7, "method": "session/prompt", "params": {
        "sessionId": "s1", "prompt": [{"type": "image", "data": "fixture", "mimeType": "image/png"},
        {"type": "text", "text": "unchanged task"}]}}
    result = transform_request(message, "/custom/agent")
    assert result["params"]["prompt"][1:] == message["params"]["prompt"]
    assert result["params"]["sessionId"] == "s1" and result["id"] == 7
    assert "/custom/agent/bin/cad-python" in result["params"]["prompt"][0]["text"]
    other = {"method": "session/new", "params": {"mcpServers": [{"name": "benchmark"}]}}
    assert transform_request(other, "/agent") is other
    assert transform_request({"method": "session/cancel"}, "/agent")["method"] == "session/cancel"


def test_acp_config_keeps_model_effort_and_auth_independent():
    env = child_environment("/agent", {"HARBOR_ACP_REQUESTED_MODEL": "openai/gpt-6.1-sol",
        "CAD_AGENT_REASONING_EFFORT": "high"})
    config = json.loads(env["CODEX_CONFIG"])
    assert config["model"] == "gpt-6.1-sol" and config["model_reasoning_effort"] == "high"
    assert "model_providers" not in config  # Direct mode uses the provider's normal endpoint.
    assert env["CODEX_PATH"] == "/agent/toolchain/node_modules/.bin/codex"
    gateway = child_environment("/agent", {"HOSTED_INFERENCE_URL": "http://localhost:1234/v1"})
    assert json.loads(gateway["CODEX_CONFIG"])["model_providers"]["harbor"]["base_url"] == "http://localhost:1234/v1"


def test_entries_keep_independent_models_and_efforts(tmp_path):
    shared = {"PATH": "/usr/bin"}
    entries = [("gpt-6.1-terra", "low"), ("gpt-6.1-terra", "high"),
               ("gpt-6.1-sol", "medium")]
    results = []
    for model, effort in entries:
        source = dict(shared, HARBOR_ACP_REQUESTED_MODEL="openai/" + model,
                      CAD_AGENT_REASONING_EFFORT=effort)
        results.append(child_environment("/agent", source))
        assert "CODEX_CONFIG" not in source
        native = CodexText2Cad(logs_dir=tmp_path / f"{model}-{effort}",
                              model_name="openai/" + model, reasoning_effort=effort)
        assert native.model_name == "openai/" + model
        assert native.options.reasoning_effort == effort
    assert [(json.loads(e["CODEX_CONFIG"])["model"],
             json.loads(e["CODEX_CONFIG"])["model_reasoning_effort"]) for e in results] == entries
    assert shared == {"PATH": "/usr/bin"}


@pytest.mark.parametrize("effort", ["", "LOW", "ultra", "unknown"])
def test_hosted_invalid_effort_fails_without_fallback(effort):
    with pytest.raises(ValueError, match="Unsupported CAD_AGENT_REASONING_EFFORT"):
        child_environment("/agent", {"CAD_AGENT_REASONING_EFFORT": effort})


def test_hosted_manifest_matches_installed_schema():
    manifest = Path(__file__).parents[1] / "harbor-agent.json"
    assert AcpSourceManifest.model_validate_json(manifest.read_text()).id == "codex-text2cad-build123d"


def test_smoke_is_not_use_and_changed_asset_invalidates_chain(tmp_path):
    assert verify_generation(tmp_path / "missing", tmp_path / "asset.usd")["status"] == "unverified"
    source, asset, step = (tmp_path / n for n in ("reconstruct.py", "asset.usd", "base.step"))
    for p in (source, asset, step):
        p.write_text(p.name)
    import hashlib
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {"kind": "cad_generation", "bundle": bundle_identity(), "source_path": str(source),
        "source_sha256": digest(source), "asset_path": str(asset), "asset_sha256": digest(asset),
        "parts": [{"step_path": str(step), "step_sha256": digest(step)}]}
    path = tmp_path / "generation.json"
    path.write_text(json.dumps(receipt))
    assert verify_generation(path, asset)["status"] == "artifact_chain_verified"
    asset.write_text("changed")
    assert verify_generation(path, asset)["status"] == "unverified"
    receipt["kind"] = "headless_smoke"
    path.write_text(json.dumps(receipt))
    assert verify_generation(path, asset)["status"] == "unverified"
