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
    public = json.loads(hub.files['runs/run-a/results.json'])
    assert public['results'][0]['fidelity']['components']['shape']['score_percent'] == 25
    assert public['results'][0]['simready']['score_percent'] == 100
    assert public['evaluation_recoveries'][0]['authoring_unchanged'] is True
    assert len(public['evaluation_recoveries']) == 1
    assert any(name.startswith('runs/run-a/evaluation-recoveries/') for name in hub.files)
    publisher(report, evaluation_recovery=request)
    assert len(json.loads(hub.files['runs/run-a/results.json'])['evaluation_recoveries']) == 1


@pytest.mark.parametrize('field', ['candidate', 'traces', 'simready', 'input_tokens', 'reasoning_effort'])
def test_repair_cannot_change_any_authoring_evidence(publication, field):
    manifest, report, hub, row, save = publication
    publisher = HuggingFacePublisher('group/results', 'run-a', manifest, api=hub)
    save([row()])
    publisher(report)
    previous = json.loads(hub.files['runs/run-a/results.json'])['results']
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
    previous = json.loads(hub.files['runs/run-a/results.json'])['results']
    request = _request(row(), 'head-1')
    if damage == 'revision':
        request['previous_revision'] = 'head-0'
    elif damage == 'source':
        request['source_tree_sha256'] = 'unknown'
    else:
        request['condition_ids'] = []
    with pytest.raises(ValueError, match='Explicit evaluation recovery'):
        _evaluation_recovery(previous, previous, request, 'head-1')
