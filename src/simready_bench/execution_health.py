"""Stop paid execution after repeated infrastructure failures or lost progress."""
from __future__ import annotations

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
        if not self.opened:
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
