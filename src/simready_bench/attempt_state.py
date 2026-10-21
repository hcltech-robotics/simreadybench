"""Durable condition ownership, phase retries and immutable measurements."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import sqlite3
import time


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class AttemptStore:
    """Commit attempt state and its audit transition in one transaction."""

    def __init__(self, output, manifest):
        self.path = Path(output).absolute() / 'attempt-state.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or self.path.exists() and self.path.stat().st_nlink != 1:
            raise ValueError('A regular attempt state database is required')
        digest = hashlib.sha256(_json(manifest).encode()).hexdigest()
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS identity(manifest_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts(
                    attempt_id TEXT PRIMARY KEY, condition_id TEXT NOT NULL,
                    owner TEXT NOT NULL, phase TEXT NOT NULL,
                    measurement TEXT NOT NULL, infrastructure TEXT NOT NULL,
                    publication TEXT NOT NULL, cleanup TEXT NOT NULL,
                    evidence TEXT NOT NULL, updated_at REAL NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS condition_owner
                    ON attempts(condition_id) WHERE cleanup != 'deleted';
                CREATE TABLE IF NOT EXISTS measurements(
                    condition_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL,
                    sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS retries(
                    attempt_id TEXT NOT NULL, phase TEXT NOT NULL,
                    used INTEGER NOT NULL, maximum INTEGER NOT NULL,
                    deadline_at REAL NOT NULL, PRIMARY KEY(attempt_id, phase));
                CREATE TABLE IF NOT EXISTS transitions(
                    sequence INTEGER PRIMARY KEY, attempt_id TEXT NOT NULL,
                    event TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at REAL NOT NULL);
                CREATE TRIGGER IF NOT EXISTS immutable_transition_update
                    BEFORE UPDATE ON transitions BEGIN
                    SELECT RAISE(ABORT, 'append_only_transitions'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_transition_delete
                    BEFORE DELETE ON transitions BEGIN
                    SELECT RAISE(ABORT, 'append_only_transitions'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_measurement_update
                    BEFORE UPDATE ON measurements BEGIN
                    SELECT RAISE(ABORT, 'append_only_measurements'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_measurement_delete
                    BEFORE DELETE ON measurements BEGIN
                    SELECT RAISE(ABORT, 'append_only_measurements'); END;
            ''')
            db.execute('BEGIN IMMEDIATE')
            saved = db.execute('SELECT manifest_sha256 FROM identity').fetchone()
            if saved is None:
                db.execute('INSERT INTO identity VALUES (?)', (digest,))
            elif saved[0] != digest:
                raise ValueError('The attempt store manifest cannot change')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _event(db, attempt_id, event, payload):
        db.execute('INSERT INTO transitions(attempt_id,event,payload,created_at) VALUES (?,?,?,?)',
                   (attempt_id, event, _json(payload), time.time()))

    def lease(self, condition_id, attempt_id, owner, *, retirement=None):
        if not all(isinstance(value, str) and value for value in (condition_id, attempt_id, owner)):
            raise ValueError('Attempt and owner identities are required')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
            if previous is not None:
                if previous['condition_id'] != condition_id or previous['owner'] != owner:
                    raise ValueError('An existing attempt has a different owner')
                return dict(previous)
            retired = retirement is not None
            if retired and (retirement.get('executor_deleted') is not True or not retirement.get('sandbox_id')):
                raise ValueError('Historical ownership requires verified deletion')
            if not retired and db.execute('SELECT 1 FROM measurements WHERE condition_id=?', (condition_id,)).fetchone():
                raise ValueError('A measured condition cannot start fresh authoring')
            if not retired and db.execute("SELECT 1 FROM attempts WHERE condition_id=? AND cleanup!='deleted'", (condition_id,)).fetchone():
                raise ValueError('The condition is still owned by an earlier attempt')
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (attempt_id, condition_id, owner, 'retired' if retired else 'leased', 'pending', 'healthy', 'not_requested',
                        'deleted' if retired else 'pending', _json({'deletion_proof': retirement} if retired else {}), time.time()))
            self._event(db, attempt_id, 'leased', {'condition_id': condition_id, 'owner': owner})
        return self.get(attempt_id)

    def get(self, attempt_id):
        with self._db() as db:
            row = db.execute('SELECT * FROM attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown attempt')
        value = dict(row)
        value['evidence'] = json.loads(value['evidence'])
        return value

    def update(self, attempt_id, **changes):
        permitted = {
            'phase': {'leased', 'preparing', 'authoring', 'sealed', 'evaluating', 'collecting', 'collected', 'retiring', 'retired', 'held'},
            'infrastructure': {'healthy', 'retrying', 'failed', 'held'},
            'publication': {'not_requested', 'pending', 'published', 'held'},
        }
        for key, value in changes.items():
            if key != 'evidence' and (key not in permitted or value not in permitted[key]):
                raise ValueError('Invalid attempt state transition')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown attempt')
            if row['publication'] == 'published' and changes.get('publication', 'published') != 'published':
                raise ValueError('Published delivery cannot move backwards')
            evidence = json.loads(row['evidence'])
            evidence.update(changes.pop('evidence', {}))
            changes.update(evidence=_json(evidence))
            if all(row[key] == value for key, value in changes.items()):
                return
            changes['updated_at'] = time.time()
            db.execute('UPDATE attempts SET ' + ','.join(key + '=?' for key in changes) + ' WHERE attempt_id=?',
                       (*changes.values(), attempt_id))
            self._event(db, attempt_id, 'updated', changes)

    def measure(self, attempt_id, sha256):
        if not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('A measurement SHA-256 is required')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown attempt')
            previous = db.execute('SELECT * FROM measurements WHERE condition_id=?', (row['condition_id'],)).fetchone()
            if previous:
                if previous['sha256'] != sha256 or previous['attempt_id'] != attempt_id:
                    raise ValueError('An immutable measurement cannot change')
                return
            db.execute('INSERT INTO measurements VALUES (?,?,?)', (row['condition_id'], attempt_id, sha256))
            db.execute("UPDATE attempts SET measurement='measured',updated_at=? WHERE attempt_id=?", (time.time(), attempt_id))
            self._event(db, attempt_id, 'measured', {'sha256': sha256})

    def release(self, attempt_id, proof):
        if not isinstance(proof, dict) or proof.get('executor_deleted') is not True or not proof.get('sandbox_id'):
            raise ValueError('Verified exact-resource deletion is required')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM attempts WHERE attempt_id=?', (attempt_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown attempt')
            evidence = json.loads(row['evidence'])
            existing_id = evidence.get('sandbox_id')
            if existing_id and existing_id != proof['sandbox_id']:
                raise ValueError('The deletion proof changes the owned resource')
            evidence.update(sandbox_id=proof['sandbox_id'], deletion_proof=proof)
            if row['cleanup'] != 'deleted':
                db.execute("UPDATE attempts SET cleanup='deleted',phase='retired',evidence=?,updated_at=? WHERE attempt_id=?",
                           (_json(evidence), time.time(), attempt_id))
                self._event(db, attempt_id, 'released', proof)

    def retry(self, attempt_id, phase, *, maximum_retries, deadline_at, now=None):
        now = time.time() if now is None else now
        if (type(maximum_retries) is not int or maximum_retries < 0
                or not isinstance(deadline_at, (int, float)) or not math.isfinite(deadline_at)
                or phase not in {'collection', 'cleanup', 'publication', 'evaluation', 'observation'}):
            raise ValueError('A bounded phase retry policy is required')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM attempts WHERE attempt_id=?', (attempt_id,)).fetchone():
                raise ValueError('Unknown attempt')
            row = db.execute('SELECT * FROM retries WHERE attempt_id=? AND phase=?', (attempt_id, phase)).fetchone()
            if row and (row['maximum'] != maximum_retries or row['deadline_at'] != deadline_at):
                raise ValueError('The saved phase retry policy cannot change')
            used = row['used'] if row else 0
            permitted = used < maximum_retries and now < deadline_at
            db.execute('INSERT OR REPLACE INTO retries VALUES (?,?,?,?,?)',
                       (attempt_id, phase, used + int(permitted), maximum_retries, deadline_at))
            self._event(db, attempt_id, 'phase_retry' if permitted else 'phase_exhausted',
                        {'phase': phase, 'used': used + int(permitted), 'maximum': maximum_retries, 'deadline_at': deadline_at})
            return permitted

    def snapshot(self):
        with self._db() as db:
            rows = [dict(row) for row in db.execute('SELECT * FROM attempts ORDER BY attempt_id')]
            measurements = {row['condition_id']: row['sha256'] for row in db.execute('SELECT * FROM measurements')}
            transitions = db.execute('SELECT COUNT(*) FROM transitions').fetchone()[0]
        for row in rows:
            row['evidence'] = json.loads(row['evidence'])
        return {'schema': 'simready-bench-attempt-state-v1', 'attempts': rows, 'measurements': measurements, 'transitions': transitions}

    def completion(self, expected):
        state = self.snapshot()
        expected = set(expected)
        measured = expected & state['measurements'].keys()
        cleanup = [row for row in state['attempts'] if row['cleanup'] != 'deleted']
        pending = [row for row in state['attempts'] if row['condition_id'] in measured and row['measurement'] == 'measured' and row['publication'] not in {'published', 'not_requested'}]
        held = [row for row in state['attempts'] if row['infrastructure'] == 'held']
        return {'planned': len(expected), 'measured': len(measured), 'publication_pending': len(pending),
                'cleanup_pending': len(cleanup), 'held_attempts': len(held),
                'complete': measured == expected and not cleanup and not pending and not held}


def read_status(output):
    """Read progress from the durable store without creating state or taking ownership."""
    root = Path(output).absolute()
    path = root / 'attempt-state.sqlite3'
    manifest = json.loads((root / 'experiment.json').read_text(encoding='utf-8'))
    from .resume import condition_plan
    expected = set(condition_plan(manifest))
    if path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('The run has no regular durable attempt store')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)
    try:
        db.row_factory = sqlite3.Row
        identity = db.execute('SELECT manifest_sha256 FROM identity').fetchone()
        if not identity or identity[0] != hashlib.sha256(_json(manifest).encode()).hexdigest():
            raise ValueError('The status manifest differs from the durable run')
        attempts = [dict(row) for row in db.execute('SELECT * FROM attempts ORDER BY attempt_id')]
        measured = {row[0] for row in db.execute('SELECT condition_id FROM measurements')}
    finally:
        db.close()
    pending = [row for row in attempts if row['measurement'] == 'measured' and row['publication'] not in {'published', 'not_requested'}]
    cleanup = [row for row in attempts if row['cleanup'] != 'deleted']
    held = [row for row in attempts if row['infrastructure'] == 'held']
    return {'schema': 'simready-bench-status-v1', 'planned': len(expected), 'measured': len(expected & measured),
            'publication_pending': len(pending), 'cleanup_pending': len(cleanup), 'held_attempts': len(held),
            'complete': expected <= measured and not pending and not cleanup and not held,
            'attempts': attempts}
