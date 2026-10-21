"""Retire interrupted bootstrap without inventing usage or restoring budget."""
import hashlib
import json
import time

import pytest

from simready_bench import resume
from test_resume import stopped_run, write

pytestmark = pytest.mark.skipif(__import__('os').name == 'nt', reason='Linux authority ledger')


def prepare(tmp_path, monkeypatch):
    output, selected, policy, ledger = stopped_run(tmp_path, monkeypatch)
    delivery_path = output / 'daytona-delivery.json'
    delivery = json.loads(delivery_path.read_text())
    delivery.update(run_id='saved-run', terminal_confirmed=True, cleanup_confirmed=True,
                    controller_exit=137, termination_reason='inference_transport_circuit_open',
                    health_policy={'schema':'simready-bench-executor-health-v1'})
    proof = {'schema':'simready-bench-executor-terminal-v1', 'sandbox_id':delivery['sandbox_id'],
             'run_id':delivery['run_id'], 'observed_at':time.time(), 'controller_exit':137,
             'reason':delivery['termination_reason'], 'all_containers_stopped':True}
    path = output / 'executor-terminal-proof.json'
    write(path, proof)
    path.chmod(0o600)
    delivery['executor_terminal_proof_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(delivery_path, delivery)
    (output/'daytona-executor/controller-status.json').unlink()
    (output/'daytona-executor/authority/parent-allowances.sqlite3').unlink()
    return output, selected, policy, ledger


def test_missing_ledger_charges_original_ceiling_and_is_idempotent(tmp_path, monkeypatch):
    output, selected, policy, ledger = prepare(tmp_path, monkeypatch)
    options = dict(policy=policy, manifest=selected, client=resume._daytona_client())
    proof = resume._retire(output, resume._reports(output), **options)
    assert proof['requests'] == 3200
    assert float(proof['maximum_spend_usd']) == 250
    assert proof['usage_status'] == 'unavailable_full_original_ceiling_charged'
    assert len(proof['unreported_admitted_conditions']) == 100
    assert proof['authority_sha256'] is None
    assert proof['executor_terminal_proof']['all_containers_stopped']
    assert resume._retire(output, resume._reports(output), **options) == proof


@pytest.mark.parametrize('fault', ['missing_proof', 'changed_proof', 'live_worker', 'unconfirmed_cleanup'])
def test_missing_ledger_cannot_retire_without_terminal_deletion_proof(tmp_path, monkeypatch, fault):
    output, selected, policy, ledger = prepare(tmp_path, monkeypatch)
    client = resume._daytona_client()
    if fault == 'missing_proof':
        (output/'executor-terminal-proof.json').unlink()
    elif fault == 'changed_proof':
        (output/'executor-terminal-proof.json').write_text('{}')
    elif fault == 'live_worker':
        client = type('Live', (), {'get': lambda self, identity: object()})()
    else:
        path = output/'daytona-delivery.json'
        delivery = json.loads(path.read_text())
        delivery['cleanup_confirmed'] = False
        write(path, delivery)
    with pytest.raises(ValueError):
        resume._retire(output, resume._reports(output), policy=policy, manifest=selected, client=client)
    assert not (output/'retirement.json').exists()
