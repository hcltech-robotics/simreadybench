"""Stop paid execution after repeated infrastructure failures or lost progress."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .recovery import _save


class ExecutionHealth:
    def __init__(self, output, *, failure_limit=5, progress_seconds=3900, now=None):
        from .resume import _read

        self.path = Path(output) / 'execution-health.json'
        self.now = now or time.time
        self.failure_limit = failure_limit
        self.progress_seconds = progress_seconds
        if self.path.exists():
            self.state = _read(self.path)
        else:
            baseline = []
            repair = Path(output) / 'executor-repair.json'
            if repair.exists():
                baseline = _read(repair).get('baseline_attempts', [])
            self.state = {'schema': 'simready-bench-execution-health-v1',
                'seen_attempts': baseline, 'failures_since_completion': 0,
                'last_completed_at': self.now(), 'status': 'healthy'}

    @property
    def opened(self):
        return self.state['status'] == 'circuit_open'

    def require_dispatch(self, allowed_pending, workers):
        if self.state['status'].startswith('repair_probe'):
            probe = self.state['repair_probe']
            if (allowed_pending is None or set(allowed_pending) != set(probe['conditions'])
                    or not 1 <= workers <= len(probe['conditions'])):
                raise ValueError('An open-circuit repair permits only its bound probe conditions')

    def begin_repair_probe(self, conditions, *, evidence_path):
        """Retain the open incident and counters while testing one verified repair."""
        from .executor_repair import repair_chain
        from .resume import _read

        conditions = sorted(set(conditions))
        chain = repair_chain(self.path.parent)
        if any(record.get('requires_new_run_spec') for _, record, _ in chain):
            raise ValueError('A scientific runtime revision requires a new run specification')
        evidence_path = Path(evidence_path).resolve(strict=True)
        evidence = evidence_path.read_bytes()
        if (not self.opened or not 1 <= len(conditions) <= 2
                or chain[-1][1]['approved_at'] <= self.state['opened_at']
                or not isinstance(json.loads(evidence), dict)):
            raise ValueError('A verified repair after circuit opening is required')
        for path in self.path.parent.glob('executions/*/daytona-delivery.json'):
            if (_read(path).get('cleanup_confirmed') is not True
                    or _read(path.parent / 'retirement.json').get('executor_deleted') is not True):
                raise ValueError('Close owned executors before a repair probe')
        folder = self.path.parent / 'execution-recoveries'
        folder.mkdir(exist_ok=True)
        receipt = folder / (chain[-1][2] + '.json')
        if receipt.exists():
            raise ValueError('This repair has already used its bounded probe')
        probe = {'conditions': conditions, 'repair_approval_sha256': chain[-1][2],
                 'evidence_path': str(evidence_path), 'evidence_sha256': hashlib.sha256(evidence).hexdigest(),
                 'started_at': self.now(), 'completed': [], 'failed_attempts': []}
        _save(receipt, {'schema': 'simready-bench-execution-recovery-v1',
                        'previous_state': self.state, 'probe': probe})
        self.state = {**self.state, 'status': 'repair_probe', 'repair_probe': probe}
        _save(self.path, self.state)

    def finish_repair_probe(self):
        """Release dispatch only after both measured probe conditions completed."""
        if self.state['status'] != 'repair_probe_passed':
            raise ValueError('The repair probe has not passed')
        self.state.update(status='healthy')
        _save(self.path, self.state)

    def observe(self, outcomes, *, active):
        seen = set(self.state['seen_attempts'])
        fresh = {row['attempt_id']: row for row in outcomes
            if row['attempt_id'] not in seen
            and row['status'] in ('completed', 'infrastructure_error')}
        successes = sum(row['status'] == 'completed' for row in fresh.values())
        failures = sum(row['status'] == 'infrastructure_error' for row in fresh.values())
        if successes:
            self.state['last_completed_at'] = self.now()
            self.state['failures_since_completion'] = 0
        self.state['failures_since_completion'] += failures
        self.state['seen_attempts'] = sorted(seen | fresh.keys())
        if self.state['status'].startswith('repair_probe'):
            probe = self.state['repair_probe']
            probe['failed_attempts'].extend(key for key, row in fresh.items()
                                             if row['status'] == 'infrastructure_error')
            completed = {key.rsplit('/', 1)[-1] for key, row in fresh.items()
                         if row['status'] == 'completed'}
            probe['completed'] = sorted(set(probe['completed']) | (completed & set(probe['conditions'])))
            if failures or active and self.now() - probe['started_at'] >= self.progress_seconds:
                self.state.update(status='circuit_open', reason='repair_probe_failed', opened_at=self.now())
            elif set(probe['completed']) == set(probe['conditions']):
                self.state['status'] = 'repair_probe_passed'
        elif not self.opened:
            reason = None
            if self.state['failures_since_completion'] >= self.failure_limit:
                reason = 'repeated_infrastructure_failures'
            elif (active and self.state['failures_since_completion']
                  and self.now() - self.state['last_completed_at'] >= self.progress_seconds):
                reason = 'no_completed_result_progress'
            if reason:
                self.state.update(status='circuit_open', reason=reason, opened_at=self.now())
        self.state['checked_at'] = self.now()
        _save(self.path, self.state)
        return self.opened
