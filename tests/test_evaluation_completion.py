"""A finished actor cannot conceal missing evaluator or capture work."""

from copy import deepcopy

from simready_bench.completion import evaluation_completion, completion_summary
from simready_bench.reporting import compact_report, render_html
import pytest


def row():
    return {
        'status': 'completed',
        'simready': {'score_percent': 100},
        'fidelity': {'status': 'measured', 'alignment': {'scale': 1}, 'components': {
            'shape': {'status': 'measured', 'score_percent': 100},
            'materials': {'appearance': {'status': 'measured', 'score_percent': 100,
                                        'earned': 1, 'eligible': 1, 'measured': 1, 'coverage_fraction': 1}},
            'kinematics': {'status': 'not_applicable', 'score_percent': None},
        }},
        'illustrations': {'status': 'rendered', 'renderer': 'storm-llvmpipe'},
        'images': [{'view': role + ' / front', 'path': role + '.png'} for role in ('reference', 'candidate')],
    }


def test_failed_comparison_requires_evaluation_recovery_without_reauthoring():
    value = row()
    value['fidelity'] = {'status': 'unknown', 'error_type': 'RuntimeError'}
    state = evaluation_completion(value, captures_enabled=True, views=['front'])
    assert not state['complete']
    assert state['authoring_complete']
    assert state['recovery_scope'] == 'evaluation_only'
    assert state['missing'] == ['fidelity_worker']  # Captures exist, worker is still required.
    value['evaluation_completion'] = state
    summary = completion_summary([value], 1)
    assert summary == {'authored_conditions': 1, 'evaluated_conditions': 0, 'benchmark_complete': False}


def test_missing_capture_blocks_completion_but_empty_candidate_is_explicitly_inapplicable():
    value = row()
    assert evaluation_completion(value, captures_enabled=True, views=['front'])['complete']
    value['images'].pop()
    assert evaluation_completion(value, captures_enabled=True, views=['front'])['missing'] == ['captures']
    value['fidelity']['alignment'] = None
    value['fidelity']['components']['shape']['score_percent'] = 0
    state = evaluation_completion(value, captures_enabled=True, views=['front'])
    assert state['complete']
    assert state['captures'] == 'not_applicable_no_alignable_candidate'


def test_applicable_kinematics_cannot_be_unknown_and_disabled_work_is_explicit():
    value = row()
    value['fidelity']['components']['kinematics']['status'] = 'unknown'
    assert evaluation_completion(value)['missing'] == ['kinematic']
    assert evaluation_completion(value, fidelity_enabled=False)['complete']


def test_report_retains_and_shows_distinct_measurement_progress():
    value = row()
    value['evaluation_completion'] = {'complete': False, 'missing': ['captures'], 'recovery_scope': 'evaluation_only'}
    summary = {'schema': 'simready-bench-portable-report-v1',
               'results': [value], 'completed_conditions': 1, 'planned_conditions': 1,
               **completion_summary([value], 1)}
    compact = compact_report(deepcopy(summary))
    assert compact['evaluated_conditions'] == 0
    assert not compact['benchmark_complete']
    assert compact['results'][0]['evaluation_completion']['missing'] == ['captures']
    html = render_html(compact)
    assert '0 of 1 evaluations complete' in html
    assert 'Missing evaluation' in html


def test_current_worker_failure_cannot_enter_leaderboard():
    from test_leaderboard import run
    from simready_bench.leaderboard import leaderboard_entry

    manifest, result = run()
    result['results'][0]['fidelity'] = {'status': 'unknown', 'error_type': 'RuntimeError'}
    with pytest.raises(ValueError, match='Every applicable fidelity measurement'):
        leaderboard_entry(result, manifest=manifest, results_url='https://example.org/results')


def test_omitted_fidelity_cannot_bypass_admission_or_runtime_defaults():
    from test_leaderboard import run
    from simready_bench.leaderboard import leaderboard_entry

    manifest, result = run()
    manifest['defaults'].pop('fidelity')
    result['results'][0].pop('fidelity')
    with pytest.raises(ValueError, match='Every applicable fidelity measurement'):
        leaderboard_entry(result, manifest=manifest, results_url='https://example.org/results')
