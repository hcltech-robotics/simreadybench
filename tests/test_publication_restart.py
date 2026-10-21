import json

from simready_bench.orchestration import ReportIndex
from test_parallel import manifest
from test_resume import portable
from simready_bench.resume import condition_plan


def test_published_readout_is_not_recommitted_after_restart(tmp_path):
    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    identity, condition = next(iter(condition_plan(selected).items()))
    batch = tmp_path / 'executions' / '001'
    batch.mkdir(parents=True)
    (batch / 'experiment.json').write_text(json.dumps(selected))
    path = portable(batch / 'attempts' / 'final', [(identity, condition, 'completed')])
    delivered = []

    def publish(report, **options):
        delivered.append(json.loads(__import__('pathlib').Path(report['results']).read_text())['completed_conditions'])

    first = ReportIndex(tmp_path, selected, publish)
    first.backfilling = False
    first.add(batch, {'directory': str(path.parent), 'results': str(path)})
    assert delivered == [1]
    restarted = ReportIndex(tmp_path, selected, publish)
    restarted.backfill({batch: selected})
    restarted.aggregate(final=True)
    assert delivered == [1]
