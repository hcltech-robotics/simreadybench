import json
from copy import deepcopy

import pytest

from simready_bench.execution_health import ExecutionHealth
from simready_bench.executor_repair import approve_executor_repair
from test_executor_repair import repack, wheel


def stopped_run(tmp_path):
    clock = [1.0]
    health = ExecutionHealth(tmp_path, now=lambda: clock[0])
    outcomes = [{'attempt_id': f'{i}/old', 'status': 'infrastructure_error'} for i in range(5)]
    assert health.observe(outcomes, active=False)
    original = wheel(tmp_path / 'original.whl')
    replacement = repack(tmp_path / 'replacement.whl', original)
    approve_executor_repair(tmp_path, original, replacement, reason='Tested repair')
    evidence = tmp_path / 'verified-repair.json'
    evidence.write_text(json.dumps({'tests_passed': True}))
    clock[0] = 2
    return health, evidence, clock


def test_probe_preserves_counters_and_restricts_dispatch_until_measurements(tmp_path):
    health, evidence, _ = stopped_run(tmp_path)
    before = deepcopy(health.state)
    health.begin_repair_probe(['a', 'b'], evidence_path=evidence)
    assert all(health.state[key] == value for key, value in before.items() if key != 'status')
    receipt = json.loads(next((tmp_path / 'execution-recoveries').glob('*.json')).read_text())
    assert receipt['previous_state'] == before
    health.require_dispatch({'a', 'b'}, 2)
    for allowed, workers in ((None, 24), ({'a', 'b'}, 24), ({'a', 'c'}, 2)):
        with pytest.raises(ValueError, match='bound probe'):
            health.require_dispatch(allowed, workers)
    with pytest.raises(ValueError, match='not passed'):
        health.finish_repair_probe()
    assert not health.observe([{'attempt_id': '6/a', 'status': 'completed'}], active=True)
    assert health.state['status'] == 'repair_probe'
    assert health.state['failures_since_completion'] == 0
    assert not health.observe([{'attempt_id': '7/b', 'status': 'completed'}], active=False)
    assert health.state['status'] == 'repair_probe_passed'
    with pytest.raises(ValueError):
        health.require_dispatch(None, 24)
    health.finish_repair_probe()
    health.require_dispatch(None, 24)
    assert set(before['seen_attempts']) <= set(health.state['seen_attempts'])


def test_any_new_probe_failure_reopens_circuit_without_reusing_approval(tmp_path):
    health, evidence, clock = stopped_run(tmp_path)
    health.begin_repair_probe(['a', 'b'], evidence_path=evidence)
    assert health.observe([{'attempt_id': '6/a', 'status': 'infrastructure_error'}], active=True)
    assert health.state['failures_since_completion'] == 6
    assert health.state['repair_probe']['failed_attempts'] == ['6/a']
    clock[0] = 3
    with pytest.raises(ValueError):
        health.begin_repair_probe(['a', 'b'], evidence_path=evidence)


def test_stalled_probe_cannot_run_indefinitely(tmp_path):
    health, evidence, clock = stopped_run(tmp_path)
    health.begin_repair_probe(['a', 'b'], evidence_path=evidence)
    clock[0] += 3901
    assert health.observe([], active=True)
    assert health.state['failures_since_completion'] == 5


def test_scientific_revision_cannot_rearm_the_frozen_run(tmp_path):
    health, evidence, _ = stopped_run(tmp_path)
    changed = wheel(tmp_path / 'scientific-revision.whl', transport=b'changed budget')
    approve_executor_repair(tmp_path, tmp_path / 'original.whl', changed,
        reason='Declared tool contract revision', append=True, scientific_revision='bounded-tools-v2')
    previous = deepcopy(health.state)
    with pytest.raises(ValueError, match='new run specification'):
        health.begin_repair_probe(['a', 'b'], evidence_path=evidence)
    assert health.state == previous
    assert not (tmp_path / 'execution-recoveries').exists()
