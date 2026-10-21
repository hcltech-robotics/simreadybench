"""Approve a transport repair while retaining every evaluator byte and attempt."""
from __future__ import annotations

import ast
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
        pricing = 'simready_bench/engine/hf_pricing.py'
        allowed = set(REPAIR_MODULES)
        if pricing in changed and pricing in before.namelist() and pricing in after.namelist():
            _pricing_fetch_only(before.read(pricing), after.read(pricing))
            allowed.add(pricing)
        if not changed or not set(changed) <= allowed:
            raise ValueError('Executor repair changes a protected evaluator or protocol file')
        return changed


def _pricing_fetch_only(before, after):
    """Permit retry orchestration while retaining every pricing validation statement."""
    old, new = ast.parse(before), ast.parse(after)
    def protected(tree):
        return [ast.dump(node) for node in tree.body
                if not (isinstance(node, ast.AsyncFunctionDef)
                        and node.name in {'fetch_hf_route', '_fetch_hf_route_once'})
                and not (isinstance(node, ast.Import)
                         and [(alias.name, alias.asname) for alias in node.names] == [('asyncio', None)])]
    old_fetch = next(node for node in old.body if getattr(node, 'name', None) == 'fetch_hf_route')
    new_fetch = next(node for node in new.body if getattr(node, 'name', None) == '_fetch_hf_route_once')
    if (protected(old) != protected(new)
            or [ast.dump(node) for node in old_fetch.body[2:]]
            != [ast.dump(node) for node in new_fetch.body[1:]]):
        raise ValueError('Executor repair changes protected HF pricing validation')


def repair_chain(output):
    """Validate append-only approvals, including every preceding package identity."""
    output = Path(output).absolute()
    paths = [output / 'executor-repair.json', *sorted((output / 'executor-repairs').glob('*.json'))]
    chain, previous_hash, original_hash = [], None, None
    for index, path in enumerate(paths):
        if path.resolve() != path or path.stat().st_nlink != 1:
            raise ValueError('A regular executor repair approval is required')
        if index and path.name != f'{index + 1:03d}.json':
            raise ValueError('Executor repair history is incomplete')
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get('schema') != 'simready-bench-executor-repair-v1':
            raise ValueError('Invalid executor repair approval')
        original, old_hash = _wheel(record['original_path'])
        replacement, new_hash = _wheel(record['replacement_path'])
        if (old_hash != record['original_sha256'] or new_hash != record['replacement_sha256']
                or _changes(original, replacement) != record['changed_modules']
                or record.get('protected_files_identical') is not True
                or index and (record.get('previous_approval_sha256') != previous_hash
                              or old_hash != original_hash)):
            raise ValueError('Executor repair differs from its recorded approval')
        previous_hash = hashlib.sha256(raw).hexdigest()
        original_hash = old_hash
        chain.append((path, record, previous_hash))
    return chain


def approve_executor_repair(output, original, replacement, *, reason, append=False):
    """Record an explicitly authorised infrastructure-only runtime replacement."""
    output = Path(output).absolute()
    path = output / 'executor-repair.json'
    if path.exists() and not append:
        raise FileExistsError('Preserving an existing executor repair approval')
    chain = repair_chain(output) if path.exists() else []
    if append and not chain:
        raise ValueError('An appended repair requires the original approval')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('An explicit operator repair reason is required')
    original, original_hash = _wheel(original)
    replacement, replacement_hash = _wheel(replacement)
    changed = _changes(original, replacement)
    if chain and original_hash != chain[0][1]['original_sha256']:
        raise ValueError('An appended repair must retain the original wheel')
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
    if chain:
        record['previous_approval_sha256'] = chain[-1][2]
        record['previous_replacement_sha256'] = chain[-1][1]['replacement_sha256']
        folder = output / 'executor-repairs'
        folder.mkdir(exist_ok=True)
        path = folder / f'{len(chain) + 1:03d}.json'
        if path.exists():
            raise FileExistsError('Preserving an existing executor repair approval')
    _save(path, record)
    return record


def repaired_wheel(output, saved_hashes, supplied=None):
    chain = repair_chain(output)
    allowed = {chain[0][1]['original_sha256']} | {record['replacement_sha256'] for _, record, _ in chain}
    if not saved_hashes <= allowed:
        raise ValueError('Executor repair differs from its recorded approval')
    if supplied is not None and _wheel(supplied)[1] not in allowed:
        raise ValueError('The supplied executor differs from the approved repair')
    return Path(chain[-1][1]['replacement_path'])
