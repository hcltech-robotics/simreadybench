import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('sync_results', Path(__file__).parents[1] / 'scripts/sync_published_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def published(monkeypatch, *, runs, evaluator='2.2.0'):
    pin = 'a' * 40
    requests = []
    data = {'formal_fidelity_primary': True, 'evaluator_version': '2.2.0', 'job_ids': ['job'], 'runs': runs}
    def read(url):
        requests.append(url)
        if url.endswith('/api/datasets/hcltech-robotics/simreadybench'):
            return json.dumps({'sha': pin}).encode()
        assert '/resolve/' + pin + '/' in url
        if url.endswith('README.md'):
            return b'[JSON](https://huggingface.co/datasets/hcltech-robotics/simreadybench/resolve/main/results/harbor/job/presentation/leaderboard.json)'
        if url.endswith('artifact-export-summary.json'):
            return json.dumps({'evaluator_version': evaluator, 'task_rows': sum(r['final_slots'] for r in runs)}).encode()
        if url.endswith('.csv'):
            return b'configuration,formal_percent\nold,100\nnew,86\n'
        return json.dumps(data).encode()
    monkeypatch.setattr(module, 'read', read)
    return requests, pin


def test_refresh_resolves_one_commit_and_keeps_existing_published_results(tmp_path, monkeypatch):
    (tmp_path / 'leaderboard.json').write_text(json.dumps({'runs': [{'id': 'old'}]}))
    runs = [{'id': i, 'job_id': 'job', 'final_slots': 300} for i in ['old', 'new']]
    requests, pin = published(monkeypatch, runs=runs)
    data = module.sync(tmp_path)
    assert [r['id'] for r in data['runs']] == ['old', 'new']
    assert data['source_revision'] == pin
    assert sum('/api/datasets/' in url for url in requests) == 1
    assert (tmp_path / 'leaderboard.json').read_bytes() == (tmp_path / 'pareto.json').read_bytes()


def test_refresh_refuses_to_remove_a_published_configuration(tmp_path, monkeypatch):
    original = json.dumps({'runs': [{'id': 'old'}]})
    (tmp_path / 'leaderboard.json').write_text(original)
    published(monkeypatch, runs=[{'id': 'new', 'job_id': 'job', 'final_slots': 300}])
    with pytest.raises(ValueError, match='disappear'):
        module.sync(tmp_path)
    assert (tmp_path / 'leaderboard.json').read_text() == original


def test_refresh_refuses_an_incompatible_artifact_inventory(tmp_path, monkeypatch):
    published(monkeypatch, runs=[{'id': 'new', 'job_id': 'job', 'final_slots': 300}], evaluator='2.1.0')
    with pytest.raises(ValueError, match='Incompatible'):
        module.sync(tmp_path)
    assert not (tmp_path / 'leaderboard.json').exists()
