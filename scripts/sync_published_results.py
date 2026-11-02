#!/usr/bin/env python3
"""Refresh the project page from the dataset card's published leaderboard."""
import argparse
import json
from pathlib import Path
import re
from urllib.request import urlopen

REPO = 'hcltech-robotics/simreadybench'
BASE = f'https://huggingface.co/datasets/{REPO}'


def read(url):
    with urlopen(url, timeout=90) as response:
        return response.read()


def sync(output, revision='main'):
    # Resolve main once: every file in a deployment comes from one commit.
    if revision == 'main':
        revision = json.loads(read(f'https://huggingface.co/api/datasets/{REPO}'))['sha']
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('A full dataset commit is required')
    card = read(f'{BASE}/resolve/{revision}/README.md').decode()
    match = re.search(r'\[JSON\]\(https://huggingface.co/datasets/' + re.escape(REPO)
                      + r'/resolve/[^/]+/(results/harbor/[^/]+/presentation/leaderboard.json)\)', card)
    if not match:
        raise ValueError('No published leaderboard JSON linked in the dataset card')
    source = match[1]
    data = json.loads(read(f'{BASE}/resolve/{revision}/{source}'))
    runs = data.get('runs', [])
    if not data.get('formal_fidelity_primary') or not runs or len({r['id'] for r in runs}) != len(runs):
        raise ValueError('Invalid leaderboard')
    previous = output / 'leaderboard.json'
    if previous.exists():
        prior = json.loads(previous.read_text())
        if not {r['id'] for r in prior['runs']}.issubset({r['id'] for r in runs}):
            raise ValueError('Published results would disappear from the project page')
    jobs = set(data['job_ids'])
    for job in jobs:
        summary = json.loads(read(f'{BASE}/resolve/{revision}/results/harbor/{job}/artifact-export-summary.json'))
        if summary['evaluator_version'] != data['evaluator_version']:
            raise ValueError('Incompatible evaluator versions')
        if summary['task_rows'] != sum(r['final_slots'] for r in runs if r['job_id'] == job):
            raise ValueError('Artifact inventory differs from leaderboard')
    data['source_revision'] = revision
    data['publication_source'] = f'{BASE}/blob/{revision}/{source}'
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, indent=2, allow_nan=False) + '\n'
    for filename in ['leaderboard.json', 'pareto.json']:
        (output / filename).write_text(encoded)
    (output / 'leaderboard.csv').write_bytes(read(f'{BASE}/resolve/{revision}/{source.rsplit("/", 1)[0]}/leaderboard.csv'))
    print(f'Loaded {len(runs)} configurations from dataset {revision}')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', type=Path, default=Path('site/assets'))
    args = parser.parse_args()
    sync(args.output, args.revision)
