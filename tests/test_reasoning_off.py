"""Explicit reasoning off survives public configuration and native Pi setup."""

import pytest

from simready_bench.benchmark import compose_experiment
from simready_bench.config import resolve_experiment
from simready_bench.engine.native_output import authoring_seconds
from simready_bench.harnesses import agent_config
from simready_bench.models import configure_reasoning


@pytest.mark.parametrize("reasoning", [False, True])
def test_explicit_off_preserves_model_capability(reasoning):
    model = {"metadata": {"reasoning": reasoning, "compat": {"supportsReasoningEffort": False}}}
    configure_reasoning(model, "off")
    assert model["metadata"]["reasoning"] is reasoning
    assert model["generation"]["reasoning_effort"] == "off"
    with pytest.raises(ValueError, match="does not support"):
        configure_reasoning(model, "medium")


def test_off_is_frozen_in_identity_and_native_thinking(setup):
    root, original, policy = setup
    policy["limits"].update(inference_requests=64, agent_seconds=authoring_seconds(64))
    conditions = []
    for effort in ("medium", "off"):
        model = dict(original["models"][0])
        configure_reasoning(model, effort)
        manifest = compose_experiment(original, model=model, repetitions=1, max_output_tokens=4096)
        condition = resolve_experiment(manifest, base_dir=root, policy=policy)["conditions"][0]
        conditions.append(condition)
        assert condition["settings"]["generation"]["reasoning_effort"] == effort
        native = agent_config(condition, bridge_id="fixture", skill_dirs=[])
        assert native["kwargs"]["thinking"] == effort
        assert native["kwargs"]["max_turns"] == 64
        assert native["kwargs"]["model_metadata"]["reasoning"] is True
    assert conditions[0]["configuration_sha256"] != conditions[1]["configuration_sha256"]


def test_public_cli_freezes_explicit_off(monkeypatch, tmp_path):
    from simready_bench.cli import main
    from simready_bench import parallel

    captured = []
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.setattr(parallel, "run_parallel_daytona", lambda manifest, **options: captured.append((manifest, options)) or {"completed_conditions": 100, "planned_conditions": 100, "results": []})
    assert main(["benchmark", "NV25", "--model", "gpt-6.1-sol", "--deploy", "daytona", "--output-dir", str(tmp_path / "run"), "--max-output-tokens", "4096", "--reasoning", "off", "--workers", "24", "--continue-on-length"]) == 0
    manifest, options = captured[0]
    assert manifest["defaults"]["generation"]["reasoning_effort"] == "off"
    assert manifest["models"][0]["generation"]["reasoning_effort"] == "off"
    assert manifest["models"][0]["metadata"]["reasoning"] is True
    assert manifest["defaults"]["generation"]["max_output_tokens"] == 4096
    assert options["workers"] == 24
    assert manifest['authoring_protocol'].endswith('-tools-bounded-v1-continue-length-v1')


@pytest.mark.parametrize('tools', ['legacy', 'bounded-v1'])
def test_length_continuation_has_explicit_identity_and_pi_option(setup, tools):
    root, original, policy = setup
    policy['limits'].update(inference_requests=64, agent_seconds=authoring_seconds(64))
    conditions = []
    for enabled in (False, True):
        manifest = compose_experiment(original, model=original['models'][0], repetitions=1,
                                      max_output_tokens=4096, tool_protocol=tools,
                                      continue_on_length=enabled)
        condition = resolve_experiment(manifest, base_dir=root, policy=policy)['conditions'][0]
        conditions.append(condition)
        native = agent_config(condition, bridge_id='fixture', skill_dirs=[])
        assert native['kwargs'].get('continue_on_length', False) is enabled
        assert native['kwargs']['max_turns'] == 64
        assert condition['authoring_protocol'].endswith('-continue-length-v1') is enabled
    assert conditions[0]['condition_id'] != conditions[1]['condition_id']
    assert conditions[0]['configuration_sha256'] != conditions[1]['configuration_sha256']


def test_length_continuation_rejects_other_harnesses(setup):
    _, manifest, _ = setup
    with pytest.raises(ValueError, match='requires the Pi harness'):
        compose_experiment(manifest, model=manifest['models'][0], harness='opencode',
                           continue_on_length=True)


def test_resume_cannot_enable_length_continuation(tmp_path):
    import json
    from simready_bench.resume import _resume_once

    (tmp_path / 'experiment.json').write_text(json.dumps({'authoring_protocol': 'authoring-64-bounded-output-4096-v1'}))
    with pytest.raises(ValueError, match='original length continuation policy'):
        _resume_once('NV25', model='fixture', harness='pi', deploy='daytona',
                     output_dir=tmp_path, continue_on_length=True)


@pytest.mark.parametrize('value', ['unknown', None, True, 1, {}, []])
def test_model_compatibility_rejects_unknown_thinking_format(setup, value):
    root, manifest, policy = setup
    manifest['models'][0]['metadata'].setdefault('compat', {})['thinkingFormat'] = value
    with pytest.raises(ValueError, match='Unsupported thinkingFormat'):
        resolve_experiment(manifest, base_dir=root, policy=policy)


def test_glm_cli_metadata_resolves_to_native_pi_without_inference(monkeypatch, tmp_path):
    import json
    from simready_bench.cli import main
    from simready_bench import parallel

    metadata = {
        'contextWindow': 1048576, 'maxTokens': 4096,
        'input': ['text', 'image'], 'reasoning': True,
        'compat': {'supportsReasoningEffort': True, 'supportsMaxOutputTokens': True,
                   'thinkingFormat': 'zai'},
    }
    metadata_path = tmp_path / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    asset = tmp_path / 'asset.usda'
    asset.write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps({'assets': [{'id': 'fixture', 'usd': 'asset.usda', 'dependencies': []}]}))
    captured = []
    def execute(manifest, **options):
        resolved = resolve_experiment(manifest, base_dir=options['output_dir'], policy=options['policy'])
        condition = resolved['conditions'][0]
        captured.append(agent_config(condition, bridge_id='fixture', skill_dirs=[]))
        return {'completed_conditions': 1, 'planned_conditions': 1, 'results': []}
    monkeypatch.setenv('HF_TOKEN', 'fixture-only')
    monkeypatch.setattr(parallel, 'run_parallel_daytona', execute)
    assert main(['benchmark', str(reference), '--model', 'zai-org/GLM-5.3-Flash:baseten',
                 '--provider', 'huggingface', '--model-metadata', str(metadata_path),
                 '--deploy', 'daytona', '--output-dir', str(tmp_path / 'run'),
                 '--reasoning', 'off', '--max-output-tokens', '4096',
                 '--continue-on-length', '--repetitions', '1', '--workers', '24']) == 0
    native = captured[0]['kwargs']
    assert native['model_metadata']['compat']['thinkingFormat'] == 'zai'
    assert native['model_metadata']['maxTokens'] == 4096
    assert native['thinking'] == 'off'
    assert native['continue_on_length'] is True
    assert native['max_turns'] == 64
