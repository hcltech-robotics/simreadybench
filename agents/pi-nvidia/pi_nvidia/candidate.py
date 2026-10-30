"""Transactional whole-package authoring. Gate: successful build + valid OpenUSD only.

Standalone: executed by the task's Python, which already supplies usd-core.
No feedback client, scoring, geometry requirement or inference is called here.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from uuid import uuid4


def manifest(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return None
    result = {}
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[name] = {'symlink': os.readlink(path)}  # Never copy/hash a link target.
        elif path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif not path.is_dir():
            raise ValueError('A candidate contains an unsupported filesystem entry')
    return result


def valid_openusd(root):
    from pxr import Sdf, Usd
    # Fresh helper process: no stale layer from an earlier authoring attempt.
    # Empty stages and composition warnings are allowed if OpenUSD opens it.
    layer = Sdf.Layer.FindOrOpen(str(Path(root) / 'asset.usd'))
    if layer is None:
        raise ValueError('OpenUSD could not open asset.usd')
    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise ValueError('OpenUSD could not open the composed stage')
    return {'usd_valid': True, 'openusd_version': list(Usd.GetVersion()),
            'criterion': 'Sdf layer and Usd.Stage.Open succeed; empty stages are allowed'}


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def exchange(prepared, final):
    """One atomic package switch, never a remove-then-rename gap."""
    if not final.exists() and not final.is_symlink():
        os.replace(prepared, final)
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(prepared), -100, os.fsencode(final), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return True


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class Candidates:
    def __init__(self, runtime):
        runtime = Path(runtime)
        self.final = Path(json.loads(runtime.read_text())['submission_entrypoint']).absolute().parent
        self.logs = runtime.parent.parent.parent
        self.store = self.logs / 'pi/candidates'
        self.verified = self.store / 'verified'
        self.verified.mkdir(parents=True, exist_ok=True)
        self.pointer = self.store / 'current.json'

    def current(self):
        if not self.pointer.exists():
            return None
        data = json.loads(self.pointer.read_text())
        snapshot = self.verified / data['id']
        if manifest(snapshot) != data['manifest']:
            raise ValueError('Preserved candidate integrity check failed')
        return snapshot, data

    def stage(self, source=None):
        self.final.parent.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix='.pi-candidate-', dir=self.final.parent))
        if source is not None:
            shutil.copytree(source, path, symlinks=True, dirs_exist_ok=True)
        return path

    def guard(self):
        current = self.current()
        if current is None:
            if manifest(self.final):
                # Retain direct writes as a draft, not an implicitly promoted build.
                draft = self.store / ('unpromoted-' + uuid4().hex)
                shutil.copytree(self.final, draft, symlinks=True)
                remove(self.final)
                self.final.mkdir()
                return {'action': 'unpromoted_draft_retained', 'draft': str(draft), 'promoted': False}
            return {'action': 'no_promoted_candidate', 'promoted': False}
        snapshot, data = current
        if manifest(self.final) == data['manifest']:
            return {'action': 'unchanged', 'candidate_id': data['id']}
        stage = self.stage(snapshot)
        try:
            exchange(stage, self.final)
        finally:
            remove(stage)
        return {'action': 'restored_last_valid', 'candidate_id': data['id'], 'usd_valid': True}

    def build(self, command):
        self.guard()
        current = self.current()
        stage = self.stage(current[0] if current else None)
        before = manifest(self.final)
        env = dict(os.environ, PI_CANDIDATE_DIR=str(stage))
        try:
            build = subprocess.run(['/bin/bash', '-e', '-o', 'pipefail', '-c', command], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = {'build_exit_code': build.returncode, 'stdout': build.stdout[-4000:], 'stderr': build.stderr[-4000:]}
            if build.returncode != 0:
                return {**output, 'promoted': False, 'reason': 'build_failed'}
            if manifest(self.final) != before:
                return {**output, 'promoted': False, 'reason': 'build_wrote_final_instead_of_staging'}
            try:
                verification = valid_openusd(stage)
            except ImportError as error:
                return {**output, 'promoted': False, 'reason': 'openusd_unavailable', 'detail': str(error)}
            except Exception as error:
                return {**output, 'promoted': False, 'reason': 'invalid_openusd', 'detail': str(error)}
            identity = uuid4().hex
            snapshot = self.verified / identity
            shutil.copytree(stage, snapshot, symlinks=True)
            data = {'id': identity, 'manifest': manifest(snapshot), **verification}
            exchanged = exchange(stage, self.final)
            try:
                atomic_json(self.pointer, data)
            except BaseException:
                if exchanged:
                    exchange(stage, self.final)
                else:
                    os.replace(self.final, stage)
                raise
            return {**output, **verification, 'promoted': True, 'candidate_id': identity,
                    'preserved_package': str(snapshot), 'local_gate_feedback_calls': 0}
        finally:
            remove(stage)
            self.guard()  # Restore a prior candidate even if an author ignored the staging path.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('action', choices=('build', 'guard'))
    parser.add_argument('--command')
    args = parser.parse_args()
    manager = Candidates(args.runtime)
    with (manager.store / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            result = manager.build(args.command) if args.action == 'build' else manager.guard()
        except Exception as error:
            result = {'promoted': False, 'reason': 'candidate_transaction_failed', 'detail': str(error)}
        if args.action == 'build' or result.get('action') not in ('unchanged', 'no_promoted_candidate'):
            with (manager.logs / 'pi-candidate-events.jsonl').open('a') as events:
                events.write(json.dumps(result) + '\n')
            atomic_json(manager.logs / 'pi-candidate-latest.json', result)
        print(json.dumps(result))
        raise SystemExit(0 if result.get('promoted') is True or 'action' in result else 1)


if __name__ == '__main__':
    main()
