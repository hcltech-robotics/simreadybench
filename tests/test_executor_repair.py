import hashlib
import json
import zipfile

import pytest

from simready_bench.executor_repair import approve_executor_repair
from simready_bench.recovery import frozen_wheel


def wheel(path, *, transport=b'original', evaluator=b'frozen evaluator'):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('simready_bench/engine/actor_relay.py', transport)
        archive.writestr('simready_bench/engine/scoring.py', evaluator)
    return path


def test_approved_repair_keeps_original_and_new_execution_identities(tmp_path):
    original = wheel(tmp_path / 'original.whl')
    replacement = wheel(tmp_path / 'replacement.whl', transport=b'fixed')
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
    replacement = wheel(tmp_path / 'replacement.whl', transport=b'fixed')
    approve_executor_repair(tmp_path, original, replacement, reason='Repair')
    with pytest.raises(FileExistsError):
        approve_executor_repair(tmp_path, original, replacement, reason='Reset')


def test_changed_repair_package_is_rejected_on_resume(tmp_path):
    original = wheel(tmp_path / 'original.whl')
    replacement = wheel(tmp_path / 'replacement.whl', transport=b'fixed')
    approve_executor_repair(tmp_path, original, replacement, reason='Repair')
    replacement.write_bytes(b'different runtime')
    with pytest.raises(ValueError, match='recorded approval'):
        frozen_wheel(tmp_path)


def test_gateway_transport_repair_keeps_agent_prompt_protected(tmp_path):
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
    receipt = approve_executor_repair(
        tmp_path, original, replacement, reason='Reject incomplete upstream streams'
    )
    assert receipt['changed_modules'] == ['simready_bench/engine/inference_gateway.py']
