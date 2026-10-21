import sqlite3

import pytest

from simready_bench.attempt_state import AttemptStore


def test_condition_lease_survives_restart_until_verified_deletion(tmp_path):
    store = AttemptStore(tmp_path, {'run': 'fixed'})
    store.lease('condition-a', '001/condition-a', 'coordinator')
    restarted = AttemptStore(tmp_path, {'run': 'fixed'})
    with pytest.raises(ValueError, match='owned'):
        restarted.lease('condition-a', '002/condition-a', 'coordinator')
    with pytest.raises(ValueError, match='deletion'):
        restarted.release('001/condition-a', {'executor_deleted': False})
    restarted.release('001/condition-a', {'executor_deleted': True, 'sandbox_id': 'owned-a'})
    restarted.lease('condition-a', '002/condition-a', 'coordinator')
    assert len(restarted.snapshot()['attempts']) == 2


def test_measured_zero_is_immutable_and_does_not_authorise_fresh_generation(tmp_path):
    store = AttemptStore(tmp_path, {'run': 'fixed'})
    store.lease('condition-a', '001/condition-a', 'coordinator')
    store.measure('001/condition-a', 'a' * 64)
    store.release('001/condition-a', {'executor_deleted': True, 'sandbox_id': 'owned-a'})
    with pytest.raises(ValueError, match='measured'):
        store.lease('condition-a', '002/condition-a', 'coordinator')
    with pytest.raises(ValueError, match='measurement'):
        store.measure('001/condition-a', 'b' * 64)
    assert store.snapshot()['measurements'] == {'condition-a': 'a' * 64}


def test_retry_budget_and_deadline_are_persisted_across_restart(tmp_path):
    store = AttemptStore(tmp_path, {'run': 'fixed'})
    store.lease('condition-a', '001/condition-a', 'coordinator')
    assert store.retry('001/condition-a', 'collection', maximum_retries=1, deadline_at=200, now=100)
    restarted = AttemptStore(tmp_path, {'run': 'fixed'})
    assert not restarted.retry('001/condition-a', 'collection', maximum_retries=1, deadline_at=200, now=101)
    with pytest.raises(ValueError, match='policy'):
        restarted.retry('001/condition-a', 'collection', maximum_retries=2, deadline_at=200, now=102)
    store.lease('condition-b', '002/condition-b', 'coordinator')
    assert not store.retry('002/condition-b', 'collection', maximum_retries=1, deadline_at=99, now=100)


def test_journal_is_append_only_and_publication_does_not_hold_resource(tmp_path):
    store = AttemptStore(tmp_path, {'run': 'fixed'})
    store.lease('condition-a', '001/condition-a', 'coordinator')
    store.measure('001/condition-a', 'a' * 64)
    store.update('001/condition-a', publication='pending')
    store.release('001/condition-a', {'executor_deleted': True, 'sandbox_id': 'owned-a'})
    pending = store.completion({'condition-a'})
    assert pending['measured'] == 1 and pending['cleanup_pending'] == 0
    assert pending['publication_pending'] == 1 and not pending['complete']
    store.update('001/condition-a', publication='published')
    assert store.completion({'condition-a'})['complete']
    with sqlite3.connect(store.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match='append_only'):
            db.execute('DELETE FROM transitions')
    with pytest.raises(ValueError, match='manifest'):
        AttemptStore(tmp_path, {'run': 'changed'})
