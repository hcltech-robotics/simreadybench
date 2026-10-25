"""Evaluation repair retains authoring evidence and requires explicit authority."""

from copy import deepcopy
import json

import pytest

from simready_bench.publishing import (
    HuggingFacePublisher, PublicationError, _evaluation_recovery,
)
from test_publishing import publication


def _request(row, revision):
    return {
        'schema': 'simready-evaluation-recovery/v1',
        'previous_revision': revision,
        'source_commit': 'a' * 40,
        'source_tree_sha256': 'b' * 64,
        'condition_ids': [row['condition_id']],
        'reason': 'Handle missing texture inputs without aborting geometry evaluation',
    }


def test_explicit_repair_updates_fidelity_and_is_idempotent(publication):
    manifest, report, hub, row, save = publication
    original = row()
    publisher = HuggingFacePublisher('group/results', 'run-a', manifest, api=hub)
    save([original])
    publisher(report)
    changed = deepcopy(original)
    changed['fidelity']['components']['shape']['score_percent'] = 25
    save([changed])
    with pytest.raises(PublicationError, match='already published'):
        publisher(report)
    request = _request(changed, 'head-1')
    publisher(report, evaluation_recovery=request)
    public = json.loads(hub.files['results/runs/run-a/results.json'])
    assert public['results'][0]['fidelity']['components']['shape']['score_percent'] == 25
    assert public['results'][0]['simready']['score_percent'] == 100
    assert public['evaluation_recoveries'][0]['authoring_unchanged'] is True
    assert len(public['evaluation_recoveries']) == 1
    assert any(name.startswith('results/runs/run-a/evaluation-recoveries/') for name in hub.files)
    publisher(report, evaluation_recovery=request)
    assert len(json.loads(hub.files['results/runs/run-a/results.json'])['evaluation_recoveries']) == 1


def test_repair_does_not_backfill_new_alignment_into_previous_unknown_fidelity(publication):
    from test_publication_metadata import with_comparison

    manifest, report, hub, row, save = publication
    original = row()
    original['fidelity'] = {'status': 'unknown'}
    publisher = HuggingFacePublisher('group/results', 'run-a', manifest, api=hub)
    publisher(save([original]))
    changed = with_comparison(row())
    publisher(save([changed]), evaluation_recovery=_request(changed, 'head-1'))
    public = json.loads(hub.files['results/runs/run-a/results.json'])
    assert public['results'][0]['fidelity']['alignment'] == changed['fidelity']['alignment']
    assert public['evaluation_recoveries'][0]['authoring_unchanged'] is True


def test_optional_repair_requires_explicit_stage_and_preserves_mandatory_result(publication):
    manifest, report, hub, row, save = publication
    publisher = HuggingFacePublisher('group/results', 'run-a', manifest, api=hub)
    publisher(save([row()]))
    previous = json.loads(hub.files['results/runs/run-a/results.json'])['results']
    current = deepcopy(previous)
    current[0]['optional_score_percent'] = 0
    current[0]['optional_score'] = {'score_percent': 0}
    current[0]['simready']['optional_score'] = {'score_percent': 0}
    request = _request(row(), 'head-1')
    with pytest.raises(ValueError, match='cannot change authoring'):
        _evaluation_recovery(previous, current, request, 'head-1')
    request['stages'] = [{'operation': 'optional_validation', 'source_commit': 'a' * 40,
                          'source_tree_sha256': 'b' * 64}]
    assert _evaluation_recovery(previous, current, request, 'head-1')['authoring_unchanged']
    current[0]['simready']['score_percent'] = 0
    with pytest.raises(ValueError, match='cannot change authoring'):
        _evaluation_recovery(previous, current, request, 'head-1')


@pytest.mark.parametrize('field', ['candidate', 'traces', 'simready', 'input_tokens', 'reasoning_effort'])
def test_repair_cannot_change_any_authoring_evidence(publication, field):
    manifest, report, hub, row, save = publication
    publisher = HuggingFacePublisher('group/results', 'run-a', manifest, api=hub)
    save([row()])
    publisher(report)
    previous = json.loads(hub.files['results/runs/run-a/results.json'])['results']
    current = deepcopy(previous)
    current[0]['fidelity']['components']['shape']['score_percent'] = 25
    if field == 'candidate':
        current[0][field]['files'][0]['sha256'] = 'c' * 64
    elif field == 'traces':
        current[0][field][0]['sha256'] = 'c' * 64
    elif field == 'simready':
        current[0][field]['score_percent'] = 0
    elif field == 'reasoning_effort':
        current[0][field] = 'low'
    else:
        current[0][field] = 123
    with pytest.raises(ValueError, match='cannot change authoring'):
        _evaluation_recovery(previous, current, _request(row(), 'head-1'), 'head-1')


@pytest.mark.parametrize('damage', ['revision', 'source', 'conditions'])
def test_repair_rejects_unbound_authority(publication, damage):
    manifest, report, hub, row, save = publication
    publisher = HuggingFacePublisher('group/results', 'run-a', manifest, api=hub)
    save([row()])
    publisher(report)
    previous = json.loads(hub.files['results/runs/run-a/results.json'])['results']
    request = _request(row(), 'head-1')
    if damage == 'revision':
        request['previous_revision'] = 'head-0'
    elif damage == 'source':
        request['source_tree_sha256'] = 'unknown'
    else:
        request['condition_ids'] = []
    with pytest.raises(ValueError, match='Explicit evaluation recovery'):
        _evaluation_recovery(previous, previous, request, 'head-1')


def test_publisher_retains_render_status_when_checking_capture_completion(publication):
    import hashlib
    from test_publishing import _png
    from test_publication_metadata import with_comparison, rig_hash

    manifest, report, hub, row, save = publication
    manifest['defaults']['fidelity'] = {'enabled': True}
    manifest['defaults']['illustrations'] = {'renderer': 'storm-llvmpipe', 'views': ['front']}
    value = with_comparison(row())
    value['illustrations'].update(status='rendered', renderer='storm-llvmpipe')
    rig = value['illustrations']['rig']
    rig['views'] = {'front': rig['views']['front']}
    value['illustrations']['rig_sha256'] = rig_hash(rig)
    raw = _png()
    (report / 'evidence/matched.png').write_bytes(raw)
    value['images'] = [{**value['images'][0], 'view': role + ' / front',
                        'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                       for role in ('reference', 'candidate')]
    publisher = HuggingFacePublisher('group/results', 'run-a', manifest, api=hub)
    publisher(save([value]))
    public = json.loads(hub.files['results/runs/run-a/results.json'])
    assert public['results'][0]['illustrations']['status'] == 'rendered'
    assert public['results'][0]['evaluation_completion']['complete']
    assert public['evaluated_conditions'] == 1
