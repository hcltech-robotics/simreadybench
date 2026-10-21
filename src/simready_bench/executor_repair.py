"""Bind runtime changes to a complete immutable package contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import zipfile

from .recovery import _save
from .provenance import content_sha256


def _wheel(path):
    path = Path(path).absolute()
    if (path.resolve() != path or not path.is_file()
            or path.stat().st_nlink != 1 or path.suffix != '.whl'):
        raise ValueError('An executor repair requires regular immutable wheel files')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(path):
    """All installed files participate, including dependencies and entry points.

    Python module names cannot enforce a semantic boundary. Even a transport
    wrapper can return unvalidated data or mutate another module at import time.
    A compatible replacement therefore retains every installed byte. Any code
    change is declared as a scientific revision and cannot rejoin a frozen run.
    """
    files = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = item.filename
            if item.is_dir():
                continue
            if (name in files or name.startswith('/') or '\\' in name
                    or ':' in name or '..' in name.split('/')):
                raise ValueError('A repair wheel requires unique confined package files')
            files[name] = hashlib.sha256(archive.read(item)).hexdigest()
    if not files:
        raise ValueError('A repair wheel cannot have an empty package contract')
    return {'schema': 'simready-bench-wheel-contract-v1', 'files': files,
            'sha256': content_sha256(files)}


def _changes(original, replacement, *, scientific_revision=None):
    before, after = _contract(original)['files'], _contract(replacement)['files']
    changed = sorted(name for name in before.keys() | after.keys()
                     if before.get(name) != after.get(name))
    if changed and not scientific_revision:
        raise ValueError('Executor repair changes a protected evaluator or protocol file. '
                         'An explicit scientific revision and new run specification are required')
    return changed


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
        if record.get('schema') not in ('simready-bench-executor-repair-v1',
                                        'simready-bench-executor-repair-v2'):
            raise ValueError('Invalid executor repair approval')
        original, old_hash = _wheel(record['original_path'])
        replacement, new_hash = _wheel(record['replacement_path'])
        if old_hash != record['original_sha256'] or new_hash != record['replacement_sha256']:
            raise ValueError('Executor repair differs from its recorded approval')
        revision = record.get('scientific_revision')
        changed = _changes(original, replacement, scientific_revision=revision)
        before, after = _contract(original), _contract(replacement)
        equivalent = not changed
        if (old_hash != record['original_sha256'] or new_hash != record['replacement_sha256']
                or changed != record['changed_modules']
                or record.get('protected_files_identical') is not equivalent
                or record.get('schema') == 'simready-bench-executor-repair-v2' and (
                    record.get('protocol_equivalent') is not equivalent
                    or record.get('requires_new_run_spec') is not (not equivalent)
                    or record.get('original_contract') != before
                    or record.get('replacement_contract') != after
                    or record.get('original_contract_sha256') != before['sha256']
                    or record.get('replacement_contract_sha256') != after['sha256'])
                or index and (record.get('previous_approval_sha256') != previous_hash
                              or record.get('previous_replacement_sha256') != chain[-1][1]['replacement_sha256']
                              or old_hash != original_hash)):
            raise ValueError('Executor repair differs from its recorded approval')
        previous_hash = hashlib.sha256(raw).hexdigest()
        original_hash = old_hash
        chain.append((path, record, previous_hash))
    return chain


def approve_executor_repair(output, original, replacement, *, reason, append=False,
                            scientific_revision=None):
    """Record a byte-equivalent repack or an explicitly separate scientific revision."""
    output = Path(output).absolute()
    path = output / 'executor-repair.json'
    if path.exists() and not append:
        raise FileExistsError('Preserving an existing executor repair approval')
    chain = repair_chain(output) if path.exists() else []
    if append and not chain:
        raise ValueError('An appended repair requires the original approval')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('An explicit operator repair reason is required')
    if scientific_revision is not None and (
            not isinstance(scientific_revision, str) or not scientific_revision.strip()):
        raise ValueError('An explicit nonempty scientific revision is required')
    original, original_hash = _wheel(original)
    replacement, replacement_hash = _wheel(replacement)
    changed = _changes(original, replacement, scientific_revision=scientific_revision)
    before, after = _contract(original), _contract(replacement)
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
    record = {'schema': 'simready-bench-executor-repair-v2', 'approved_at': time.time(),
        'reason': reason, 'original_path': str(original),
        'original_sha256': original_hash, 'replacement_path': str(replacement),
        'replacement_sha256': replacement_hash, 'changed_modules': changed,
        'baseline_attempts': sorted(set(baseline)), 'protected_files_identical': not changed,
        'protocol_equivalent': not changed, 'requires_new_run_spec': bool(changed),
        'original_contract': before, 'replacement_contract': after,
        'original_contract_sha256': before['sha256'],
        'replacement_contract_sha256': after['sha256']}
    if scientific_revision is not None:
        record['scientific_revision'] = scientific_revision
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
    if any(record.get('requires_new_run_spec') for _, record, _ in chain):
        raise ValueError('A scientific runtime revision requires a new run specification')
    allowed = {chain[0][1]['original_sha256']} | {record['replacement_sha256'] for _, record, _ in chain}
    if not saved_hashes <= allowed:
        raise ValueError('Executor repair differs from its recorded approval')
    if supplied is not None and _wheel(supplied)[1] not in allowed:
        raise ValueError('The supplied executor differs from the approved repair')
    return Path(chain[-1][1]['replacement_path'])
