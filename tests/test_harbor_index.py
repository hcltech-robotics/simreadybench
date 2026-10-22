"""Asset rows retain genuine Harbor trials, rewards and confined file access."""

from datetime import datetime, timezone
import json
import os

import pytest

from simready_bench.harbor_index import refresh_asset_index


def _trial(root, number, *, asset='bucket', model='model-a', finished=True):
    from harbor.models.trial.config import TrialConfig
    from harbor.models.trial.result import AgentInfo, ModelInfo, TrialResult
    from harbor.models.task.id import LocalTaskId
    from harbor.models.verifier.result import VerifierResult

    condition = {'condition_id': f'condition-{number:012d}', 'asset_id': asset}
    source = root / 'native-jobs' / f'simready-bench-{number:012d}'
    trial = source / f'{asset}__trial{number}'
    trial.mkdir(parents=True)
    config = TrialConfig.model_validate({
        'task': {'path': str(root / asset)},
        'agent': {'name': 'pi', 'model_name': model},
        'trial_name': trial.name, 'trials_dir': str(source),
    })
    (source / 'config.json').write_text(json.dumps({
        'agents': [{'name': 'pi', 'model_name': model}],
        'tasks': [{'path': str(root / asset)}],
    }))
    (trial / 'config.json').write_text(config.model_dump_json())
    (trial / 'trajectory.json').write_text('{"steps": []}')
    if finished:
        result = TrialResult(
            task_name=asset, trial_name=trial.name, trial_uri=trial.as_uri(),
            task_id=LocalTaskId(path=root / asset), task_checksum='a' * 64,
            config=config, agent_info=AgentInfo(name='pi', version='0.85.1',
                model_info=ModelInfo(name=model)),
            started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
            verifier_result=VerifierResult(rewards={'simready_reward': number}),
        )
        (trial / 'result.json').write_text(result.model_dump_json())
    return condition, trial


def test_asset_index_groups_models_and_repetitions_with_native_stats(tmp_path):
    from harbor.viewer.scanner import JobScanner

    conditions = [_trial(tmp_path, n, model='model-a' if n < 3 else 'model-b')[0]
                  for n in range(1, 5)]
    conditions.append(_trial(tmp_path, 5, asset='kettle', finished=False)[0])
    index = refresh_asset_index(tmp_path, conditions)
    scanner = JobScanner(index)
    assert set(scanner.list_jobs()) == {'bucket', 'kettle'}
    assert scanner.get_job_config('bucket') is not None
    bucket = scanner.get_job_result('bucket')
    assert bucket.n_total_trials == bucket.stats.n_completed_trials == 4
    assert {r.verifier_result.rewards['simready_reward'] for r in bucket.trial_results} == {1, 2, 3, 4}
    assert len(bucket.stats.evals) == 2
    assert sorted(e.metrics[0]['mean'] for e in bucket.stats.evals.values()) == [1.5, 3.5]
    assert scanner.get_job_result('kettle').stats.n_running_trials == 1
    assert scanner.get_job_result('kettle').finished_at is None
    assert len(scanner.list_trials('bucket')) == 4


def test_index_files_stay_confined_and_follow_atomic_source_replacement(tmp_path):
    condition, trial = _trial(tmp_path, 1)
    index = refresh_asset_index(tmp_path, [condition])
    linked = index / 'bucket' / trial.name / 'trajectory.json'
    assert linked.resolve().is_relative_to(index / 'bucket')
    assert os.path.samefile(linked, trial / 'trajectory.json')
    replacement = trial / 'replacement.json'
    replacement.write_text('{"steps": [1]}')
    replacement.replace(trial / 'trajectory.json')
    refresh_asset_index(tmp_path, [condition])
    assert json.loads(linked.read_text()) == {'steps': [1]}
    assert not linked.is_symlink()


def test_pending_assets_have_one_row_and_unsafe_names_are_rejected(tmp_path):
    from harbor.viewer.scanner import JobScanner

    condition = {'asset_id': 'bucket', 'condition_id': 'condition-000000000001'}
    index = refresh_asset_index(tmp_path, [condition])
    pending = JobScanner(index).get_job_result('bucket')
    assert pending.stats.n_pending_trials == 1
    assert pending.stats.n_completed_trials == 0
    with pytest.raises(ValueError, match='safe path'):
        refresh_asset_index(tmp_path, [{**condition, 'asset_id': '../outside'}])


def test_native_viewer_serves_grouped_trial_trajectory_with_path_confinement(tmp_path):
    from fastapi.testclient import TestClient
    from harbor.viewer.server import create_app

    condition, trial = _trial(tmp_path, 1)
    index = refresh_asset_index(tmp_path, [condition])
    with TestClient(create_app(index)) as client:
        assert client.get('/api/jobs').status_code == 200
        assert client.get('/api/jobs/bucket').status_code == 200
        assert client.get('/api/jobs/bucket/trials').status_code == 200
        path = f'/api/jobs/bucket/trials/{trial.name}/files/trajectory.json'
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == {'steps': []}
