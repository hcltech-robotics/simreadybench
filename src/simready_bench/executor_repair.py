"""Approve a transport repair while retaining every evaluator byte and attempt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import zipfile

from .recovery import _save

REPAIR_MODULES = {
    'simready_bench/engine/actor_relay.py',
    'simready_bench/engine/inference_gateway.py',
    'simready_bench/daytona.py', 'simready_bench/orchestration.py',
    'simready_bench/recovery.py', 'simready_bench/resume.py',
    'simready_bench/execution_health.py', 'simready_bench/executor_repair.py',
}


def _wheel(path):
    path = Path(path).absolute()
    if (path.resolve() != path or not path.is_file()
            or path.stat().st_nlink != 1 or path.suffix != '.whl'):
        raise ValueError('An executor repair requires regular immutable wheel files')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _changes(original, replacement):
    with zipfile.ZipFile(original) as before, zipfile.ZipFile(replacement) as after:
        names = set(before.namelist()) | set(after.namelist())
        changed = []
        for name in sorted(names):
            if '.dist-info/' in name:
                continue
            old = before.read(name) if name in before.namelist() else None
            new = after.read(name) if name in after.namelist() else None
            if old != new:
                changed.append(name)
        if not changed or not set(changed) <= REPAIR_MODULES:
            raise ValueError('Executor repair changes a protected evaluator or protocol file')
        return changed


def approve_executor_repair(output, original, replacement, *, reason):
    """Record an explicitly authorised infrastructure-only runtime replacement."""
    output = Path(output).absolute()
    path = output / 'executor-repair.json'
    if path.exists():
        raise FileExistsError('Preserving an existing executor repair approval')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('An explicit operator repair reason is required')
    original, original_hash = _wheel(original)
    replacement, replacement_hash = _wheel(replacement)
    changed = _changes(original, replacement)
    baseline = []
    for receipt in sorted(output.glob('executions/*/retirement.json')):
        value = json.loads(receipt.read_text())
        if value.get('executor_deleted') is not True:
            raise ValueError('Close old executors before approving a transport repair')
        baseline.extend(receipt.parent.name + '/' + identity
            for identity in value.get('admitted_conditions', []))
    pending = [p for p in output.glob('executions/*/daytona-delivery.json')
        if not json.loads(p.read_text()).get('cleanup_confirmed')]
    if pending:
        raise ValueError('Close old executors before approving a transport repair')
    record = {'schema': 'simready-bench-executor-repair-v1', 'approved_at': time.time(),
        'reason': reason, 'original_path': str(original),
        'original_sha256': original_hash, 'replacement_path': str(replacement),
        'replacement_sha256': replacement_hash, 'changed_modules': changed,
        'baseline_attempts': sorted(set(baseline)), 'protected_files_identical': True}
    _save(path, record)
    return record


def repaired_wheel(output, saved_hashes, supplied=None):
    path = Path(output) / 'executor-repair.json'
    if path.resolve() != path or path.stat().st_nlink != 1:
        raise ValueError('A regular executor repair approval is required')
    record = json.loads(path.read_text())
    if record.get('schema') != 'simready-bench-executor-repair-v1':
        raise ValueError('Invalid executor repair approval')
    original, old_hash = _wheel(record['original_path'])
    replacement, new_hash = _wheel(record['replacement_path'])
    if (old_hash != record['original_sha256'] or new_hash != record['replacement_sha256']
            or not saved_hashes <= {old_hash, new_hash}
            or _changes(original, replacement) != record['changed_modules']
            or record.get('protected_files_identical') is not True):
        raise ValueError('Executor repair differs from its recorded approval')
    if supplied is not None and _wheel(supplied)[1] not in {old_hash, new_hash}:
        raise ValueError('The supplied executor differs from the approved repair')
    return replacement
