"""Local historical migration preserves bytes and accounts for absent sources."""
import hashlib
import importlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from simready_bench.publishing import build_viewer_update

REVISION = '0d19347b23ed31a077f045ca3454bacc5bed1d26'


def migration():
    return importlib.import_module('simready_bench.results_migration')


def archive(tmp_path, shards=11, rows=606):
    root = tmp_path / 'archive'
    root.mkdir()
    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    put('.gitattributes', b'*.parquet filter=lfs diff=lfs merge=lfs -text\n')
    put('LICENSE', b'Code license; assets retain upstream terms.\n')
    put('README.md', b'---\nlicense: mit\nconfigs:\n- config_name: all\n  default: true\n  data_files: viewer/run-0/scores.parquet\n---\n# Historical results\n![Chart](media/chart.png)\n[Gallery](gallery/index.html)\n')
    put('media/chart.png', b'unchanged png')
    put('gallery/index.html', b'<img src="../runs/run-0/evidence/capture.png">')
    put('runs/run-0/evidence/capture.png', b'capture')
    put('runs/run-0/scores.csv', b'condition_id,simready_reward\na,42.25\n')
    put('runs/run-0/results.json', b'{"provenance":{"revision":"original"},"results":[{"simready_reward":42.25}]}')
    put('benchmark/task.json', b'{"task":"clean"}')
    index = {'runs': {}}
    for n in range(shards):
        count = rows // shards + (n < rows % shards)
        stream = pa.BufferOutputStream()
        pq.write_table(pa.table({'condition_id': [f'{n}-{i}' for i in range(count)],
                                 'simready_reward': [42.25] * count}), stream)
        put(f'viewer/run-{n}/scores.parquet', stream.getvalue().to_pybytes())
        index['runs'][f'run-{n}'] = {'scores': f'runs/run-{n}/scores.csv',
            'viewer_scores': f'viewer/run-{n}/scores.parquet',
            'groups': {'by_model': [{'split': 'm', 'path': f'viewer/run-{n}/scores.parquet'}]}}
    put('viewer-index.json', json.dumps(index).encode())
    put('leaderboard.json', json.dumps({'entries': [
        {'results_url': 'https://huggingface.co/datasets/hcltech-robotics/simreadybench/blob/main/runs/run-0/report.html', 'score': 42.25},
        {'results_url': f'https://huggingface.co/datasets/hcltech-robotics/simreadybench/blob/{REVISION}/runs/run-0/report.html', 'score': 42.25}]}).encode())
    return root


def test_complete_archive_dry_run_then_additive_apply(tmp_path):
    root = archive(tmp_path)
    original = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    receipt = migration().migrate_results(root, source_revision=REVISION)
    assert not (root / 'results').exists()
    assert receipt['mode'] == 'dry-run'
    assert receipt['viewer_shards'] == 11 and receipt['viewer_rows'] == 606
    assert len(receipt['missing_indexed_sources']) == 10
    assert {e['source'] for e in receipt['files']} == set(original)
    assert receipt['source_revision'] == REVISION
    assert receipt['rollback']['delete_paths'] == []
    assert receipt['rollback']['original_revision'] == REVISION
    applied = migration().migrate_results(root, source_revision=REVISION, apply=True)
    assert applied['mode'] == 'apply'
    assert all((root / name).read_bytes() == raw for name, raw in original.items())
    for entry in applied['files']:
        assert entry['source_sha256'] == hashlib.sha256(original[entry['source']]).hexdigest()
        if entry['destination']:
            raw = (root / entry['destination']).read_bytes()
            assert entry['destination_sha256'] == hashlib.sha256(raw).hexdigest()
            assert entry['destination_bytes'] == len(raw)
    index = json.loads((root / 'results/viewer-index.json').read_text())
    assert index['runs']['run-10']['scores'] == 'results/runs/run-10/scores.csv'
    assert index['runs']['run-10']['groups']['by_model'][0]['path'] == 'results/viewer/run-10/scores.parquet'
    board = json.loads((root / 'results/leaderboard.json').read_text())
    assert '/blob/main/results/runs/' in board['entries'][0]['results_url']
    assert board['entries'][1] == json.loads(original['leaderboard.json'])['entries'][1]
    assert (root / 'results/runs/run-0/results.json').read_bytes() == original['runs/run-0/results.json']
    assert (root / 'results/media/chart.png').read_bytes() == original['media/chart.png']
    assert (root / 'results/gallery/index.html').read_bytes() == original['gallery/index.html']
    metadata = yaml.safe_load((root / 'results/README.md').read_text().split('---\n')[1])
    assert metadata['configs'][0]['data_files'] == 'results/viewer/run-0/scores.parquet'
    assert not (root / 'results/benchmark').exists()
    # Repeated local apply is a verification, with no replacement or deletion.
    assert migration().migrate_results(root, source_revision=REVISION, apply=True)['viewer_rows'] == 606


def test_inventory_counts_can_be_parameterized_for_synthetic_archives(tmp_path):
    root = archive(tmp_path, shards=2, rows=3)
    with pytest.raises(ValueError, match='shard|row'):
        migration().migrate_results(root, source_revision=REVISION)
    receipt = migration().migrate_results(root, source_revision=REVISION, expected_shards=2, expected_rows=3)
    assert receipt['viewer_rows'] == 3


@pytest.mark.parametrize('damage', ['revision', 'symlink', 'conflict', 'unsafe_index', 'missing_shard'])
def test_unsafe_or_incomplete_migrations_fail_before_copy(tmp_path, damage):
    root = archive(tmp_path)
    revision = REVISION
    if damage == 'revision':
        revision = 'a' * 40
    elif damage == 'symlink':
        (root / 'runs/run-0/link').symlink_to('/etc/passwd')
    elif damage == 'conflict':
        (root / 'results/media').mkdir(parents=True)
        (root / 'results/media/chart.png').write_bytes(b'wrong')
    elif damage == 'unsafe_index':
        index = json.loads((root / 'viewer-index.json').read_text())
        index['runs']['run-0']['scores'] = '../outside.csv'
        (root / 'viewer-index.json').write_text(json.dumps(index))
    else:
        (root / 'viewer/run-10/scores.parquet').unlink()
    with pytest.raises(ValueError):
        migration().migrate_results(root, source_revision=revision, apply=True)
    assert not (root / 'results/viewer-index.json').exists()


def test_explicit_root_card_composition_defaults_to_tasks_and_names_history():
    module = importlib.import_module('simready_bench.publishing')
    assert hasattr(module, 'compose_benchmark_card'), 'Explicit benchmark card composition is missing'
    existing = '---\nlicense: other\nconfigs:\n- config_name: tasks\n  data_files: benchmark/tasks.parquet\n---\n# Benchmark\nTask asset licenses remain upstream.\n'
    text = module.compose_benchmark_card(existing)
    metadata = yaml.safe_load(text.split('---\n')[1])
    assert metadata['configs'] == [{'config_name': 'tasks', 'data_files': 'benchmark/tasks.parquet', 'default': True}]
    assert metadata['license'] == 'other'
    assert text.endswith('# Benchmark\nTask asset licenses remain upstream.\n')


def test_project_results_and_gallery_use_same_current_dataset():
    root = Path(__file__).parents[2]
    data = json.loads((root / 'site/assets/pareto.json').read_text())
    gallery = json.loads((root / 'site/assets/explorer/manifest.json').read_text())
    assert 'assets/pareto.json' in (root / 'site/app.js').read_text()
    assert all(variant['run_id'] in data['job_ids'] for obj in gallery['objects'] for variant in obj['variants'])
    assert all(variant['trial_id'] in variant['download_url'] for obj in gallery['objects'] for variant in obj['variants'])
    assert 'artifact-export-summary.json' in (root / '.github/workflows/pages.yml').read_text()


def test_root_card_retains_optional_history_when_no_new_index_is_supplied():
    from simready_bench.publishing import compose_benchmark_card
    existing = ('---\nlicense: other\nconfigs:\n'
                '- config_name: tasks\n  data_files: benchmark/tasks.parquet\n'
                '- config_name: historical_all\n  default: true\n  data_files: results/viewer/old/scores.parquet\n'
                '---\n# Benchmark\n')
    metadata = yaml.safe_load(compose_benchmark_card(existing).split('---\n')[1])
    assert metadata['configs'] == [
        {'config_name': 'tasks', 'data_files': 'benchmark/tasks.parquet', 'default': True},
        {'config_name': 'historical_all', 'data_files': 'results/viewer/old/scores.parquet'}]


@pytest.mark.parametrize('data_files', ['results/viewer/old/scores.parquet', 'benchmark/../runs/scores.csv'])
def test_root_card_rejects_nonclean_default_tasks(data_files):
    from simready_bench.publishing import compose_benchmark_card
    existing = '---\nconfigs:\n- config_name: tasks\n  data_files: ' + data_files + '\n---\n# Benchmark\n'
    with pytest.raises(ValueError):
        compose_benchmark_card(existing)


def test_root_card_optional_history_configs_are_explicit_and_never_default():
    from simready_bench.publishing import compose_benchmark_card
    index, _ = build_viewer_update('old', {'results': []}, 'condition_id\n')
    existing = '---\nlicense: other\nconfigs:\n- config_name: tasks\n  data_files: benchmark/tasks.parquet\n---\n# Benchmark\n'
    metadata = yaml.safe_load(compose_benchmark_card(existing, historical_index=index).split('---\n')[1])
    assert [c['config_name'] for c in metadata['configs'] if c.get('default')] == ['tasks']
    history = [c for c in metadata['configs'] if c['config_name'].startswith('historical_')]
    assert history[0]['config_name'] == 'historical_all'
    assert history[0]['data_files'][0]['path'] == ['results/viewer/old/scores.parquet']


def test_tracked_inventory_rejects_an_incomplete_local_archive(tmp_path):
    root = archive(tmp_path)
    inventory = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
    receipt = migration().migrate_results(root, source_revision=REVISION, tracked_paths=inventory)
    assert receipt['inventory_complete_against_tracked_paths'] is True
    (root / 'media/chart.png').unlink()
    with pytest.raises(ValueError, match='tracked-path inventory'):
        migration().migrate_results(root, source_revision=REVISION, tracked_paths=inventory, apply=True)
    assert not (root / 'results').exists()
