"""Exercise repository skill commands through the public CLI without execution."""

import importlib
from pathlib import Path
import re
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / '.agents/skills'


def documented_commands(skill):
    document = (SKILLS / skill / 'SKILL.md').read_text()
    return [shlex.split(block.replace('\\\n', ''))
            for block in re.findall(r'```sh\n(.*?)```', document, re.S)
            if block.strip().startswith('simready-bench benchmark ')]


def offline_supervisor(monkeypatch):
    from simready_bench import supervision

    calls = []

    def supervise(argv, *, output_dir, maximum_restarts=5):
        calls.append({'argv': list(argv), 'output_dir': output_dir, 'maximum_restarts': maximum_restarts})
        arguments = []
        source = iter(argv)
        for value in source:
            if value == '--supervise':
                continue
            if value == '--coordinator-restarts':
                next(source)
                continue
            arguments.append(value)
        from simready_bench.cli import main

        return main(arguments)

    monkeypatch.setattr(supervision, 'supervise', supervise)
    return calls


@pytest.mark.parametrize('skill,reference,repetitions', [
    ('run-simreadybench-custom', './references.json', 2),
    ('run-simreadybench-nv25', 'NV25', 4),
])
def test_skill_examples_dispatch_public_cli_and_preserve_protocol(skill, reference, repetitions, monkeypatch):
    workflow = importlib.import_module('simready_bench.benchmark')
    captured = []

    def execute(selected_reference, **options):
        captured.append((selected_reference, options))
        return {'completed_conditions': 100, 'planned_conditions': 100, 'results': []}

    monkeypatch.setattr(workflow, 'benchmark', execute)
    supervised = offline_supervisor(monkeypatch)
    from simready_bench.cli import main

    commands = documented_commands(skill)
    assert len(commands) == 2
    for command in commands:
        assert command[0] == 'simready-bench'
        assert main(command[1:]) == 0
    assert len(supervised) == 2
    assert all(call['maximum_restarts'] == 5 for call in supervised)
    selected_reference, options = captured[0]
    assert selected_reference == reference
    assert options['authoring_turns'] == 64
    assert options['deploy'] == 'daytona'
    assert options['repetitions'] == repetitions
    assert options['workers'] == 4 and options['harness'] == 'pi'
    assert options['skills'] is None
    assert options['maximum_spend_usd'] is None
    assert options['model'] == 'gpt-6.1-sol' and options['provider'] == 'openai'
    selected_reference, custom = captured[1]
    assert selected_reference == ('NV25' if skill.endswith('nv25') else './references/hinged-case/asset.usd')
    assert custom['model'] == 'custom-vlm'
    assert custom['policy'] == Path('custom-model-policy.json')
    assert custom['authoring_turns'] == 64 and custom['deploy'] == 'daytona'
    assert custom['repetitions'] == (4 if skill.endswith('nv25') else 1)
    assert options['output_dir'] != custom['output_dir']


def test_custom_skill_options_reach_public_cli_on_resume(monkeypatch):
    workflow = importlib.import_module('simready_bench.benchmark')
    captured = {}

    def execute(reference, **options):
        captured.update(reference=reference, **options)
        return {'completed_conditions': 6, 'planned_conditions': 6, 'results': []}

    monkeypatch.setattr(workflow, 'benchmark', execute)
    supervised = offline_supervisor(monkeypatch)
    command = documented_commands('run-simreadybench-custom')[0]
    command += ['--authoring-turns', '16', '--harness', 'opencode', '--workers', '2',
                '--provider', 'huggingface', '--model', 'example/model:provider',
                '--model-metadata', './model.json', '--skills', './actor-skills/usd',
                '--hf-repo', 'example/results', '--hf-run', 'custom-model-opencode-turn16',
                '--coordinator-restarts', '3', '--resume']
    from simready_bench.cli import main

    assert main(command[1:]) == 0
    assert captured['authoring_turns'] == 16 and captured['harness'] == 'opencode'
    assert captured['provider'] == 'huggingface' and captured['model'] == 'example/model:provider'
    assert captured['model_metadata'] == Path('model.json')
    assert captured['skills'] == [Path('actor-skills/usd')]
    assert captured['workers'] == 2 and captured['resume'] is True
    assert captured['output_dir'] == Path('../runs/custom-sol')
    assert captured['hf_repo'] == 'example/results'
    assert captured['hf_run'] == 'custom-model-opencode-turn16'
    assert supervised[0]['maximum_restarts'] == 3
    assert supervised[0]['output_dir'] == Path('../runs/custom-sol')


def test_official_skill_uses_the_packaged_public_reference_aliases():
    from simready_bench.benchmark import reference_manifest

    official, _, package_inputs = reference_manifest('NV25')
    alias, _, _ = reference_manifest('nvidia-25')
    assert official == alias and package_inputs is False
    assert len(official['assets']) == 25
    assert official['benchmark_freeze']['id'] == 'nvidia-25'
    assert set(official['source']['packages']) == {asset['id'] for asset in official['assets']}
