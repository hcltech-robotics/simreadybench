"""Group retained native Harbor trials by reference asset for its viewer."""

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import NAMESPACE_URL, uuid5


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n')
    temporary.replace(path)


def _link_trial(source, target):
    """Keep real trial files inside the viewer's confinement boundary."""
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob('*'):
        if path.is_symlink() or not path.is_file():
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination.exists() and os.path.samefile(path, destination):
                continue
            temporary = destination.with_name(destination.name + '.link-tmp')
            temporary.unlink(missing_ok=True)
            os.link(path, temporary)
            temporary.replace(destination)
        except FileNotFoundError:
            # A running Harbor trial can atomically replace a file while scanned.
            continue


def refresh_asset_index(root, conditions):
    """Index actual trial identities and results without changing execution files.

    Every asset has one job row. Its models, harnesses and repetitions remain
    separate native trials with their original UUIDs, rewards and trajectories.
    The index uses hard links so image and trajectory endpoints remain confined
    to the displayed job directory without copying large artefacts.
    """
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult

    root = Path(root)
    destination = root / 'harbor-assets'
    destination.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for condition in conditions:
        asset = condition['asset_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,190}', asset):
            raise ValueError('Harbor asset names must be safe path components')
        groups[asset].append(condition)
    now = datetime.now(timezone.utc)
    for asset, planned in groups.items():
        group = destination / asset
        group.mkdir(exist_ok=True)
        results, agents, tasks, starts, updates = [], [], [], [], []
        active = 0
        for condition in planned:
            source = root / 'native-jobs' / ('simready-bench-' + condition['condition_id'][-12:])
            try:
                config = json.loads((source / 'config.json').read_text())
            except (OSError, ValueError):
                continue
            for agent in config.get('agents', []):
                if agent not in agents:
                    agents.append(agent)
            for task in config.get('tasks', []):
                if task not in tasks:
                    tasks.append(task)
            for trial in source.iterdir():
                if not trial.is_dir() or not (trial / 'config.json').is_file():
                    continue
                _link_trial(trial, group / trial.name)
                try:
                    result = TrialResult.model_validate_json((trial / 'result.json').read_text())
                except (OSError, ValueError):
                    active += 1
                    starts.append(datetime.fromtimestamp((trial / 'config.json').stat().st_mtime, timezone.utc))
                    continue
                starts.append(result.started_at or now)
                updates.append(result.finished_at or result.started_at or now)
                if result.finished_at is None:
                    active += 1
                else:
                    results.append(result)
        # JobStats is Harbor's own aggregation over the actual native results.
        stats = JobStats.from_trial_results(results, n_total_trials=len(planned), n_running_trials=active)
        complete = len(results) == len(planned)
        result = JobResult(
            id=uuid5(NAMESPACE_URL, str(root.resolve()) + '/' + asset),
            started_at=min(starts, default=now), updated_at=now,
            finished_at=max(updates) if complete else None,
            n_total_trials=len(planned), stats=stats, trial_results=results,
        )
        _write(group / 'config.json', {
            'job_name': asset, 'jobs_dir': str(destination),
            'n_attempts': len(planned), 'n_concurrent_trials': len(planned),
            'agents': agents, 'tasks': tasks,
        })
        _write(group / 'result.json', result.model_dump(mode='json'))
        _write(group / 'simready-index.json', {
            'schema': 'simready-harbor-asset-index/v1',
            'asset_id': asset,
            'condition_ids': [condition['condition_id'] for condition in planned],
            'execution_jobs_directory': str(root / 'native-jobs'),
        })
    return destination
