import hashlib
import json
import zipfile

import pytest

from simready_bench.executor_repair import approve_executor_repair
from simready_bench.recovery import frozen_wheel


def test_appended_repair_retains_and_verifies_every_previous_approval(tmp_path):
    original = wheel(tmp_path / 'original.whl')
    first = repack(tmp_path / 'first.whl', original, comment=b'first repack')
    second = repack(tmp_path / 'second.whl', original, comment=b'second repack')
    approve_executor_repair(tmp_path, original, first, reason='First verified repair')
    approval = (tmp_path / 'executor-repair.json').read_bytes()
    approve_executor_repair(tmp_path, original, second, reason='Second verified repair', append=True)
    assert (tmp_path / 'executor-repair.json').read_bytes() == approval
    for number, path in enumerate((original, first, second)):
        (tmp_path / f'parallel-plan-{number}.json').write_text(json.dumps({
            'wheel_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert frozen_wheel(tmp_path, original) == second
    assert frozen_wheel(tmp_path, first) == second
    first.write_bytes(b'changed previous runtime')
    with pytest.raises(ValueError):
        frozen_wheel(tmp_path)


def test_pricing_retry_is_a_declared_runtime_revision(tmp_path):
    from simready_bench.engine import hf_pricing
    import inspect

    current = inspect.getsource(hf_pricing)
    start = current.index('async def fetch_hf_route(')
    helper = current.index('async def _fetch_hf_route_once(')
    original = (current[:start].replace('import asyncio\n', '')
                + current[helper:].replace('async def _fetch_hf_route_once(', 'async def fetch_hf_route(', 1))
    original = original.replace('    duration = min(20, deadline_at - time.time())',
        '    if model not in HF_PRICING:\n        _deny("hf_cost_model_not_authorized")\n'
        '    duration = min(20, deadline_at - time.time())', 1)
    name = 'simready_bench/engine/hf_pricing.py'
    before, after = tmp_path / 'before.whl', tmp_path / 'after.whl'
    for path, source in ((before, original), (after, current)):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(name, source)
    with pytest.raises(ValueError, match='scientific revision'):
        approve_executor_repair(tmp_path, before, after, reason='Pricing retry')


def wheel(path, *, transport=b'original', evaluator=b'frozen evaluator'):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('simready_bench/engine/actor_relay.py', transport)
        archive.writestr('simready_bench/engine/scoring.py', evaluator)
    return path


def repack(path, original, *, comment=b'repack'):
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(path, 'w') as archive:
        for name in source.namelist():
            archive.writestr(name, source.read(name))
        archive.comment = comment
    return path


def test_approved_repair_keeps_original_and_new_execution_identities(tmp_path):
    original = wheel(tmp_path / 'original.whl')
    replacement = repack(tmp_path / 'replacement.whl', original)
    old_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    (tmp_path / 'parallel-plan-001.json').write_text(json.dumps({'wheel_sha256': old_hash}))
    approve_executor_repair(tmp_path, original, replacement, reason='Operator authorised relay repair')
    assert frozen_wheel(tmp_path, replacement) == replacement
    assert frozen_wheel(tmp_path, original) == replacement
    new_hash = hashlib.sha256(replacement.read_bytes()).hexdigest()
    (tmp_path / 'parallel-plan-002.json').write_text(json.dumps({'wheel_sha256': new_hash}))
    assert frozen_wheel(tmp_path) == replacement


def test_repair_cannot_change_scoring_or_replace_a_recorded_approval(tmp_path):
    original = wheel(tmp_path / 'original.whl')
    changed = wheel(tmp_path / 'changed.whl', evaluator=b'changed scoring')
    with pytest.raises(ValueError, match='protected evaluator'):
        approve_executor_repair(tmp_path, original, changed, reason='Repair')
    replacement = repack(tmp_path / 'replacement.whl', original)
    approve_executor_repair(tmp_path, original, replacement, reason='Repair')
    with pytest.raises(FileExistsError):
        approve_executor_repair(tmp_path, original, replacement, reason='Reset')


def test_changed_repair_package_is_rejected_on_resume(tmp_path):
    original = wheel(tmp_path / 'original.whl')
    replacement = repack(tmp_path / 'replacement.whl', original)
    approve_executor_repair(tmp_path, original, replacement, reason='Repair')
    replacement.write_bytes(b'different runtime')
    with pytest.raises(ValueError, match='recorded approval'):
        frozen_wheel(tmp_path)


def test_gateway_and_agent_prompt_are_both_protected(tmp_path):
    original = tmp_path / 'original.whl'
    replacement = tmp_path / 'replacement.whl'
    changed_prompt = tmp_path / 'changed-prompt.whl'
    for path, gateway, prompt in (
        (original, b'original transport', b'frozen prompt'),
        (replacement, b'terminal stream validation', b'frozen prompt'),
        (changed_prompt, b'terminal stream validation', b'new prompt'),
    ):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('simready_bench/engine/inference_gateway.py', gateway)
            archive.writestr('simready_bench/engine/pi.py', prompt)
    with pytest.raises(ValueError, match='protected evaluator'):
        approve_executor_repair(tmp_path, original, changed_prompt, reason='Transport repair')
    with pytest.raises(ValueError, match='scientific revision'):
        approve_executor_repair(
            tmp_path, original, replacement, reason='Reject incomplete upstream streams'
        )
