import json

from simready_bench.attempt_state import AttemptStore
from simready_bench.cli import main
from test_parallel import manifest
from simready_bench.resume import condition_plan


def test_status_reports_measurement_publication_and_cleanup_separately(tmp_path, capsys):
    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    (tmp_path / 'experiment.json').write_text(json.dumps(selected))
    identity = next(iter(condition_plan(selected)))
    store = AttemptStore(tmp_path, selected)
    store.lease(identity, '001/' + identity, 'coordinator')
    store.measure('001/' + identity, 'a' * 64)
    store.update('001/' + identity, publication='pending')
    store.release('001/' + identity, {'executor_deleted': True, 'sandbox_id': 'worker'})
    assert main(['status', '--output-dir', str(tmp_path), '--json']) == 0
    status = json.loads(capsys.readouterr().out)
    assert status['measured'] == 1
    assert status['publication_pending'] == 1
    assert status['cleanup_pending'] == 0
    assert not status['complete']
