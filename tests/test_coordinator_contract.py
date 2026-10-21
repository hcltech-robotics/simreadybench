import json
import hashlib

import pytest

from simready_bench import benchmark, daytona, orchestration, parallel, resume
from simready_bench.cli import main
from simready_bench.recovery import RecoveryLimitReached
from simready_bench.runtime_images import DEFAULT_IMAGES
from test_parallel import manifest


def test_default_cli_uses_coordinator_from_first_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture')
    output = tmp_path / 'run'

    def policy(**options):
        return {'roots': {name: str(output / name) for name in
                ('assets', 'skills', 'input_views', 'private_manifests')},
                'providers': {'selected': options['provider']}, 'limits': {}, 'runtime': {}}

    def execute(selected, **options):
        assert options['workers'] == 1
        assert selected['tool_protocol'] == 'bounded-v1'
        return {'planned_conditions': 25, 'completed_conditions': 25,
                'results': [{'asset_id': row['id'], 'status': 'completed'} for row in selected['assets']]}

    def direct(*args, **kwargs):
        pytest.fail('The initial single-worker command bypassed its coordinator')

    monkeypatch.setattr(benchmark, 'prepare_policy', policy)
    monkeypatch.setattr(parallel, 'run_parallel_daytona', execute)
    monkeypatch.setattr(daytona, 'run_daytona', direct)
    assert main(['benchmark', 'NV25', '--model', 'gpt-6.1-sol', '--deploy', 'daytona',
                 '--output-dir', str(output), '--repetitions', '1', '--max-spend-usd', '250',
                 '--max-retries', '0']) == 0


@pytest.mark.parametrize('exception', [ValueError('receipt integrity mismatch'), OSError('temporary download failure')])
def test_collection_failure_is_bounded_and_persisted(tmp_path, monkeypatch, exception):
    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    output = tmp_path / 'run'
    batch = output / 'executions' / '001'
    batch.mkdir(parents=True)
    (batch / 'experiment.json').write_text(json.dumps(selected))
    (batch / 'daytona-delivery.json').write_text(json.dumps({
        'schema': 'simready-bench-daytona-delivery-v2', 'sandbox_id': 'owned-worker',
        'collection': 'pending', 'cleanup_confirmed': False, 'terminal_confirmed': True,
    }))
    calls = []

    def collect(*args, **options):
        calls.append(1)
        if len(calls) > 2:
            pytest.fail('Collection exceeded its saved retry allowance')
        raise exception

    monkeypatch.setattr(daytona, 'reconnect_daytona', collect)
    monkeypatch.setattr(resume, '_daytona_client', object)
    monkeypatch.setattr(daytona, 'contain_executor', lambda *args, **kwargs: {'cleanup_pending': True}, raising=False)
    with pytest.raises(RecoveryLimitReached, match='collection|Collection|observation|Observation'):
        orchestration.coordinate_daytona(selected, policy={}, output_dir=output, workers=1,
            wheel=None, cpu_snapshot=None, runtime_images=DEFAULT_IMAGES,
            max_retries=0, retry_delay_seconds=0, maximum_retry_delay_seconds=0)
    assert len(calls) == 1
    from simready_bench.attempt_state import AttemptStore
    state = AttemptStore(output, selected).snapshot()
    assert state['attempts'][0]['infrastructure'] == 'held'
    assert state['attempts'][0]['cleanup'] == 'pending'
    with pytest.raises(RecoveryLimitReached):
        orchestration.coordinate_daytona(selected, policy={}, output_dir=output, workers=1,
            wheel=None, cpu_snapshot=None, runtime_images=DEFAULT_IMAGES,
            max_retries=0, retry_delay_seconds=0, maximum_retry_delay_seconds=0)
    assert len(calls) == 1


def test_saved_dispatch_resumes_its_original_batch(tmp_path, monkeypatch):
    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    roots = {name: str(tmp_path / name) for name in ('assets', 'skills', 'input_views', 'private_manifests')}
    from pathlib import Path
    for path in roots.values():
        Path(path).mkdir()
    policy = {'roots': roots}
    batch = parallel._stage_shards(selected, policy, tmp_path, 1,
        assets_root=roots['assets'], skills_root=roots['skills'], wheel=None,
        cpu_snapshot=None, runtime_images=DEFAULT_IMAGES)[0][0]
    retry = {'max_retries': 0, 'retry_delay_seconds': 0, 'maximum_retry_delay_seconds': 0}
    (tmp_path / 'coordinator-state.json').write_text(json.dumps({
        'schema': 'simready-bench-coordinator-v1',
        'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        'dispatches': {batch.name: list(resume.condition_plan(selected))},
        'observation_failures': {}, 'not_before': {}, 'retry_policy': retry}))
    dispatched = []
    def execute(*args, **options):
        dispatched.append(options['output_dir'])
        raise ValueError('Fixture terminal stop')
    monkeypatch.setattr(daytona, 'run_daytona', execute)
    monkeypatch.setattr(daytona, 'contain_executor', lambda *a, **k: {})
    monkeypatch.setattr(resume, '_daytona_client', object)
    monkeypatch.setattr(resume, '_remaining', lambda *a: (1000, 250))
    with pytest.raises(RecoveryLimitReached):
        orchestration.coordinate_daytona(selected, policy=policy, output_dir=tmp_path,
            workers=1, wheel=None, cpu_snapshot=None, runtime_images=DEFAULT_IMAGES, **retry)
    assert dispatched == [batch]
    assert len(list((tmp_path / 'executions').iterdir())) == 1


def test_measured_batch_without_delivery_cannot_author_again(tmp_path, monkeypatch):
    from test_resume import portable
    from simready_bench.attempt_state import AttemptStore
    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    identity, condition = next(iter(resume.condition_plan(selected).items()))
    batch = tmp_path / 'executions' / '001'
    batch.mkdir(parents=True)
    (batch / 'experiment.json').write_text(json.dumps(selected))
    report = portable(batch / 'attempts' / 'final', [(identity, condition, 'completed')])
    (batch / 'operator.json').write_text('{}')
    index = orchestration.ReportIndex(tmp_path, selected)
    index.backfill({batch: selected})
    index.aggregate()
    assert report.is_file()
    calls = []
    monkeypatch.setattr(daytona, 'run_daytona', lambda *a, **k: calls.append(1))
    monkeypatch.setattr(daytona, 'contain_executor', lambda *a, **k: {})
    monkeypatch.setattr(resume, '_daytona_client', object)
    with pytest.raises(RecoveryLimitReached):
        orchestration.coordinate_daytona(selected, policy={}, output_dir=tmp_path,
            workers=1, wheel=None, cpu_snapshot=None, runtime_images=DEFAULT_IMAGES,
            max_retries=0, retry_delay_seconds=0, maximum_retry_delay_seconds=0)
    state = AttemptStore(tmp_path, selected).snapshot()
    assert calls == []
    assert state['measurements'].keys() == {identity}
    assert state['attempts'][0]['infrastructure'] == 'held'


def test_failure_before_delivery_retries_the_same_owned_batch(tmp_path, monkeypatch):
    from simready_bench.attempt_state import AttemptStore
    from test_resume import portable

    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    batch = tmp_path / 'executions' / '001'
    batch.mkdir(parents=True)
    (batch / 'experiment.json').write_text(json.dumps(selected))
    (batch / 'operator.json').write_text('{}')
    dispatched, authored = [], []

    def execute(shard, **options):
        dispatched.append(options['output_dir'])
        if len(dispatched) == 1:
            raise OSError('Transient failure before executor allocation')
        authored.append(1)
        path = portable(batch / 'attempts/reference', [
            (identity, condition, 'completed')
            for identity, condition in resume.condition_plan(shard).items()
        ])
        options['on_report']({'directory': str(path.parent), 'results': str(path)})

    monkeypatch.setattr(daytona, 'run_daytona', execute)
    result = orchestration.coordinate_daytona(
        selected, policy={}, output_dir=tmp_path, workers=1,
        wheel=None, cpu_snapshot=None, runtime_images=DEFAULT_IMAGES,
        max_retries=1, retry_delay_seconds=0, maximum_retry_delay_seconds=0,
    )
    assert result['completed_conditions'] == 1
    assert dispatched == [batch, batch] and authored == [1]
    assert len(list((tmp_path / 'executions').iterdir())) == 1
    state = AttemptStore(tmp_path, selected)
    with state._db() as database:
        retry = database.execute('SELECT used,maximum,deadline_at FROM retries').fetchone()
    assert tuple(retry[:2]) == (1, 1)
    saved = json.loads((tmp_path / 'coordinator-state.json').read_text())
    assert retry['deadline_at'] == saved['observation_deadlines']['001']


@pytest.mark.parametrize('permanent', [False, True])
def test_held_data_can_retry_cleanup_without_reauthoring(tmp_path, monkeypatch, permanent):
    from simready_bench.attempt_state import AttemptStore
    monkeypatch.setattr(orchestration.time, 'time', lambda: 100.0)
    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    batch = tmp_path / 'executions' / '001'
    batch.mkdir(parents=True)
    (batch / 'experiment.json').write_text(json.dumps(selected))
    (batch / 'daytona-delivery.json').write_text(json.dumps({
        'schema': 'simready-bench-daytona-delivery-v2', 'sandbox_id': 'owned',
        'collection': 'pending', 'cleanup_confirmed': False, 'terminal_confirmed': True}))
    observed, cleaned = [], []
    def collect(*a, **k):
        observed.append(1)
        raise ValueError('Integrity failure')
    def contain(*a, **k):
        cleaned.append(1)
        if permanent or len(cleaned) == 1:
            raise OSError('Temporary cleanup failure')
        return {'executor_deleted': True, 'sandbox_id': 'owned'}
    monkeypatch.setattr(daytona, 'reconnect_daytona', collect)
    monkeypatch.setattr(daytona, 'contain_executor', contain)
    monkeypatch.setattr(resume, '_daytona_client', object)
    index_type = orchestration.ReportIndex
    constructed = []
    def index(*args):
        constructed.append(1)
        assert len(constructed) == 1, 'Held cleanup must precede rejected report loading'
        return index_type(*args)
    monkeypatch.setattr(orchestration, 'ReportIndex', index)
    for _ in range(5):
        with pytest.raises(RecoveryLimitReached):
            orchestration.coordinate_daytona(selected, policy={}, output_dir=tmp_path,
                workers=1, wheel=None, cpu_snapshot=None, runtime_images=DEFAULT_IMAGES,
                max_retries=0, retry_delay_seconds=0, maximum_retry_delay_seconds=0)
    assert len(observed) == 1
    assert len(cleaned) == (3 if permanent else 2)
    row = AttemptStore(tmp_path, selected).snapshot()['attempts'][0]
    assert row['cleanup'] == ('pending' if permanent else 'deleted')
    assert row['infrastructure'] == 'held'
