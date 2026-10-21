"""Explicit per-call output caps reach Pi, provider admission and frozen identity."""

import asyncio
from dataclasses import replace
import json

import pytest

from simready_bench.benchmark import compose_experiment, validate_native_policy_limits
from simready_bench.config import resolve_experiment
from simready_bench.engine.inference_gateway import GatewayDenied, InferenceGateway
from simready_bench.engine.native_output import authoring_protocol, authoring_seconds, authoring_turns, native_output
from simready_bench.harnesses import agent_config
from test_hf_cost import config as hf_config, MODEL
from test_gateway_stream_integrity import exercise, response_frames


def test_cap_reaches_resolved_pi_and_changes_identity(setup):
    root, original, policy = setup
    policy["limits"].update(inference_requests=64, agent_seconds=authoring_seconds(64))
    resolved = []
    for cap in (None, 8192, 16384):
        manifest = compose_experiment(original, model=original["models"][0], repetitions=1, max_output_tokens=cap)
        condition = resolve_experiment(manifest, base_dir=root, policy=policy)["conditions"][0]
        resolved.append(condition)
        assert condition["authoring_protocol"] == authoring_protocol(64, cap)
        assert authoring_turns(condition["authoring_protocol"]) == 64
        assert native_output(condition) is (cap is None)
        assert condition["settings"]["generation"]["max_output_tokens"] == cap
        assert condition["settings"]["budgets"]["max_output_tokens"] == cap
        config = agent_config(condition, bridge_id="fixture", skill_dirs=[])
        assert config["kwargs"]["max_turns"] == 64
        assert config["kwargs"]["native_output_policy"] == ("provider-native" if cap is None else "bounded")
        assert config["kwargs"]["model_metadata"]["compat"]["supportsMaxOutputTokens"] is (cap is not None)
        if cap is not None:
            assert config["kwargs"]["model_metadata"]["maxTokens"] == cap
    assert len({c["condition_id"] for c in resolved}) == 3
    assert len({c["configuration_sha256"] for c in resolved}) == 3


@pytest.mark.parametrize("cap", [0, 15, True, 8192.5, 10000001])
def test_invalid_explicit_caps_are_rejected(cap):
    with pytest.raises(ValueError, match="Output token cap"):
        authoring_protocol(64, cap)


def test_cap_cannot_widen_operator_limit():
    policy = {"limits": {"max_output_tokens": 8192}}
    validate_native_policy_limits(policy, authoring_turns=64, max_output_tokens=8192)
    with pytest.raises(ValueError, match="policy"):
        validate_native_policy_limits(policy, authoring_turns=64, max_output_tokens=16384)


def test_hf_pi_uses_max_tokens_at_the_local_gateway(setup):
    from simready_bench.models import model_configuration

    root, original, policy = setup
    policy["limits"].update(inference_requests=64, agent_seconds=authoring_seconds(64))
    policy["providers"]["selected"].update(kind="huggingface", base_url="https://router.huggingface.co/v1")
    manifest = compose_experiment(original, model=model_configuration("huggingface", MODEL), repetitions=1, max_output_tokens=8192)
    condition = resolve_experiment(manifest, base_dir=root, policy=policy)["conditions"][0]
    config = agent_config(condition, bridge_id="fixture", skill_dirs=[])
    assert config["kwargs"]["model_metadata"]["compat"]["maxTokensField"] == "max_tokens"


def test_public_cli_freezes_cap_and_reasoning(monkeypatch, tmp_path):
    from simready_bench.cli import main
    from simready_bench import parallel

    captured = []
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.setattr(parallel, "run_parallel_daytona", lambda manifest, **options: captured.append((manifest, options)) or {"completed_conditions": 100, "planned_conditions": 100, "results": []})
    assert main(["benchmark", "NV25", "--model", "gpt-6.1-sol", "--deploy", "daytona", "--output-dir", str(tmp_path / "run"), "--max-output-tokens", "8192", "--reasoning", "medium"]) == 0
    manifest, options = captured[0]
    assert manifest["authoring_protocol"] == "authoring-64-bounded-output-8192-v1-tools-bounded-v1"
    assert manifest["defaults"]["generation"] == {"output_limit": "bounded", "max_output_tokens": 8192, "reasoning_effort": "medium"}
    assert options["policy"]["limits"]["max_output_tokens"] == 8192
    allowance = json.loads((tmp_path / "run" / "allowance.json").read_text())
    assert allowance["source"] == "capped-request-capacity"
    assert allowance["max_output_tokens"] == 8192


@pytest.mark.parametrize("private_relay", [False, True])
def test_hf_gateway_enforces_cap_and_preserves_messages(private_relay):
    gateway = object.__new__(InferenceGateway)
    configuration = replace(hf_config(), max_output_tokens=8192, native_output_cap=True)
    if private_relay:
        # A trusted relay changes the host, never the Hugging Face wire protocol.
        object.__setattr__(configuration, "upstream_base_url", "http://127.0.0.1:9999/v1")
        object.__setattr__(configuration, "hf_router_relay", {"model": MODEL})
    gateway.config = configuration
    messages = [{"role": "user", "content": [{"type": "text", "text": "asset"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQID"}}]}, {"role": "assistant", "tool_calls": [{"id": "tool-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "tool-1", "content": "done"}]
    payload = {"model": MODEL, "messages": messages, "max_tokens": 8192}
    assert gateway._validate_payload(json.dumps(payload).encode()) == payload
    missing = {k: v for k, v in payload.items() if k != "max_tokens"}
    assert gateway._validate_payload(json.dumps(missing).encode()) == payload
    assert gateway._validate_payload(json.dumps({**payload, "max_tokens": 4096}).encode())["max_tokens"] == 4096
    for cap in (8193, True):
        with pytest.raises(GatewayDenied):
            gateway._validate_payload(json.dumps({**payload, "max_tokens": cap}).encode())


@pytest.mark.parametrize("api_style,terminal", [("openai-completions", "length"), ("openai-responses", "incomplete")])
def test_explicit_cap_length_is_completed_without_retry(tmp_path, monkeypatch, api_style, terminal):
    body = response_frames(api_style, terminal)
    record, received, error = asyncio.run(exercise(tmp_path, monkeypatch, api_style, body, output_cap=8192))
    assert error is None and received == body
    assert record["status"] == "completed"
    assert record["requested_output_token_cap"] == record["effective_output_token_cap"] == 8192


def test_resume_rejects_changed_cap_before_dispatch(tmp_path):
    from simready_bench.resume import _resume_once

    (tmp_path / "experiment.json").write_text(json.dumps({"defaults": {"generation": {"max_output_tokens": 8192}}}))
    with pytest.raises(ValueError, match="original output token cap"):
        _resume_once("NV25", model=MODEL, harness="pi", deploy="daytona", output_dir=tmp_path, max_output_tokens=16384)


def test_daytona_relay_allows_bounded_requests(tmp_path, monkeypatch):
    from test_daytona import test_full_executor_reserves_before_create_exports_then_deletes

    test_full_executor_reserves_before_create_exports_then_deletes(
        tmp_path, monkeypatch, False, False, None, output_cap=8192
    )


def test_declared_large_context_retains_output_cap():
    from test_custom_native_admission import spec
    from simready_bench.engine.inference_gateway import GatewayConfig

    declared = {**spec(), "context_window": 1048576}
    config = GatewayConfig(declared["base_url"], declared["model"], "fixture-only", api_style="openai-completions", model_spec=declared, max_input_tokens=1048576, max_output_tokens=8192, native_output_cap=True)
    assert config.max_input_tokens == 1048576
    assert config.max_output_tokens == 8192
