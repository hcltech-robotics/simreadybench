"""Authoring limits reach the public command, frozen identities and Pi."""

import importlib
import json
import time

import pytest

from simready_bench.benchmark import (
    compose_experiment, prepare_policy, _new_provider, native_run_allowance,
)
from simready_bench.config import condition_identity, resolve_experiment
from simready_bench.engine.native_output import (
    authoring_protocol,
    authoring_seconds,
    authoring_turns,
    native_output,
)
from simready_bench.harnesses import agent_config


@pytest.mark.parametrize("turns", [1, 12, 32, 64, 128])
def test_turn_limit_reaches_each_condition_and_pi(setup, turns):
    root, original, policy = setup
    manifest = compose_experiment(
        original, model=original["models"][0], repetitions=1, authoring_turns=turns
    )
    policy["limits"].update(
        inference_requests=turns, agent_seconds=authoring_seconds(turns)
    )
    resolved = resolve_experiment(manifest, base_dir=root, policy=policy)
    condition = resolved["conditions"][0]
    assert condition["authoring_protocol"] == authoring_protocol(turns)
    assert native_output(condition)
    assert condition["settings"]["budgets"]["inference_requests"] == turns
    assert condition["settings"]["budgets"]["agent_seconds"] == authoring_seconds(turns)
    assert condition["settings"]["generation"]["max_output_tokens"] is None
    assert condition["model"]["native_output_facts"]["protocol"] == authoring_protocol(turns)
    config = agent_config(condition, bridge_id="fixture", skill_dirs=[])
    assert config["kwargs"]["max_turns"] == turns
    assert config["kwargs"]["native_output_policy"] == "provider-native"
    assert config["override_timeout_sec"] == authoring_seconds(turns)


def test_turn_protocol_changes_identity_and_preserves_legacy_32_identity():
    ids = dict(asset="cup", harness="pi", model="selected", skill_set="none")
    old = condition_identity(
        ids, repetition=0, master_seed=0,
        authoring_protocol="authoring-32-native-output-v1",
    )
    new = condition_identity(ids, repetition=0, master_seed=0)
    assert old != new
    from simready_bench.config import _digest

    identity = {
        **ids, "repetition": 0, "master_seed": 0,
        "authoring_protocol": "authoring-32-native-output-v1",
    }
    seed = int(_digest(identity)[:16], 16) % 2**31
    assert old == ("condition-" + _digest({**identity, "seed": seed}), seed)


@pytest.mark.parametrize("turns", [0, -1, 129, True, 1.5, "64"])
def test_invalid_turn_counts_are_rejected(turns):
    with pytest.raises(ValueError):
        authoring_protocol(turns)


@pytest.mark.parametrize("protocol", ["authoring-064-native-output-v1", "bounded", ""])
def test_invalid_protocol_is_not_treated_as_native(protocol):
    with pytest.raises(ValueError):
        authoring_turns(protocol)
    assert not native_output({"authoring_protocol": protocol})


def test_provider_and_operator_allow_all_64_turns_for_nv25(tmp_path):
    provider = _new_provider(
        "openai", "gpt-6.1-sol", api_key_env=None,
        maximum_spend_usd=100, conditions=100,
    )
    assert provider["project_budget"]["maximum_requests"] == 6400
    assert provider["project_budget"]["expires_at"] >= time.time() + authoring_seconds(64)
    policy = prepare_policy(
        output_dir=tmp_path, provider=provider,
        maximum_spend_usd=100, conditions=100, deploy="daytona",
    )
    assert policy["limits"]["inference_requests"] == 64
    assert policy["limits"]["agent_seconds"] == authoring_seconds(64)
    assert policy["limits"]["max_output_tokens"] is None


@pytest.mark.parametrize("provider,model,expected", [
    ("openai", "gpt-6.1-sol", 41792),
    ("huggingface", "google/gemma-4-31B-it:novita", 905.9712),
])
def test_default_accounting_capacity_covers_all_native_turns(provider, model, expected):
    full = native_run_allowance(model, provider=provider, conditions=100)
    assert full == expected
    assert native_run_allowance(
        model, provider=provider, conditions=100, authoring_turns=32
    ) == full / 2


@pytest.mark.parametrize("explicit", [None, 250])
@pytest.mark.parametrize("relative", [False, True])
def test_public_cli_freezes_full_turn_capacity_and_honours_explicit_cap(
    monkeypatch, tmp_path, explicit, relative,
):
    from simready_bench.cli import main
    from simready_bench import daytona

    output = tmp_path / "run"
    if relative:
        (tmp_path / "checkout").mkdir()
        monkeypatch.chdir(tmp_path / "checkout")
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "private-fixture-key")

    def execute(manifest, **options):
        calls.append((manifest, options))
        return {"completed_conditions": 100, "planned_conditions": 100, "results": []}

    monkeypatch.setattr(daytona, "run_daytona", execute)
    argv = [
        "benchmark", "NV25", "--model", "gpt-6.1-sol", "--deploy", "daytona",
        "--output-dir", "../run" if relative else str(output),
    ]
    if explicit is not None:
        argv += ["--max-spend-usd", str(explicit)]
    assert main(argv) == 0
    manifest, options = calls[0]
    assert options["output_dir"] == output
    assert manifest["authoring_protocol"] == "authoring-64-native-output-v1"
    assert manifest["defaults"]["budgets"]["inference_requests"] == 64
    assert manifest["defaults"]["generation"]["max_output_tokens"] is None
    budget = options["policy"]["providers"]["selected"]["project_budget"]
    assert budget["maximum_requests"] == 6400
    assert budget["maximum_spend_usd"] == (41792 if explicit is None else explicit)
    allowance = json.loads((output / "allowance.json").read_text())
    assert allowance["source"] == ("native-request-capacity" if explicit is None else "explicit")
    assert allowance["maximum_spend_usd"] == budget["maximum_spend_usd"]
    assert "private-fixture-key" not in (output / "operator.json").read_text()


def test_cli_exposes_64_default_and_custom_turn_option(monkeypatch, tmp_path):
    from simready_bench.cli import main
    workflow = importlib.import_module("simready_bench.benchmark")
    seen = []

    def execute(reference, **options):
        seen.append(options)
        return {"completed_conditions": 1, "planned_conditions": 1, "results": []}

    monkeypatch.setattr(workflow, "benchmark", execute)
    common = ["benchmark", "NV25", "--model", "gpt-6.1-sol", "--output-dir", str(tmp_path)]
    assert main(common) == 0
    assert seen[-1]["authoring_turns"] is None
    assert main(common + ["--authoring-turns", "12"]) == 0
    assert seen[-1]["authoring_turns"] == 12


def test_explicit_policy_request_ceiling_is_preserved(monkeypatch, tmp_path):
    workflow = importlib.import_module("simready_bench.benchmark")
    policy = {
        "providers": {"one": {
            "kind": "openai", "base_url": "https://api.openai.com/v1",
            "project_budget": {"model": "gpt-6.1-sol", "maximum_spend_usd": 100},
        }},
        "limits": {"inference_requests": 32, "maximum_spend_usd": 100},
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="authoring turns exceed"):
        workflow.benchmark(
            "NV25", model="gpt-6.1-sol", deploy="daytona", output_dir=output,
            policy=path, maximum_spend_usd=100, authoring_turns=64,
        )
    assert not output.exists()


@pytest.mark.parametrize("restriction", [
    {"agent_seconds": 1800},
    {"max_input_tokens": 64000},
    {"max_output_tokens": 4096},
])
def test_preparing_native_execution_cannot_widen_explicit_policy(tmp_path, restriction):
    from simready_bench.benchmark import validate_native_policy_limits

    policy = {"limits": restriction}
    before = json.dumps(policy, sort_keys=True)
    with pytest.raises(ValueError, match="policy"):
        validate_native_policy_limits(policy, authoring_turns=64)
    with pytest.raises(ValueError, match="policy"):
        prepare_policy(
            output_dir=tmp_path, provider={}, maximum_spend_usd=100,
            conditions=1, deploy="daytona", existing=policy,
        )
    assert json.dumps(policy, sort_keys=True) == before
    assert not (tmp_path / "operator.json").exists()


def test_fractional_condition_allowance_survives_custom_matrix_preflight(tmp_path, monkeypatch):
    workflow = importlib.import_module("simready_bench.benchmark")
    from simready_bench import daytona

    granted = tmp_path / "granted"
    granted.mkdir()
    provider = _new_provider(
        "openai", "gpt-6.1-sol", api_key_env=None,
        maximum_spend_usd=100, conditions=75,
    )
    original = prepare_policy(
        output_dir=granted, provider=provider, maximum_spend_usd=100,
        conditions=75, deploy="daytona",
    )
    assert original["limits"]["maximum_condition_spend_usd"] == 1.333333
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-private-key")
    calls = []

    def execute(manifest, **options):
        calls.append(options)
        return {"completed_conditions": 75, "planned_conditions": 75, "results": []}

    monkeypatch.setattr(daytona, "run_daytona", execute)
    result = workflow.benchmark(
        "NV25", model="gpt-6.1-sol", deploy="daytona", repetitions=3,
        output_dir=tmp_path / "run", policy=granted / "operator.json",
    )
    assert result["completed_conditions"] == 75
    assert calls[0]["policy"]["limits"]["maximum_condition_spend_usd"] == 1.333333
    assert calls[0]["policy"]["providers"]["selected"]["project_budget"] == provider["project_budget"]
