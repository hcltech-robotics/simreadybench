import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('nv100_results', Path(__file__).parents[1] / 'scripts/build_nv100_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(condition, reference, score, cost, *, retry=False):
    ident = condition + reference
    return {'id': ident + ('retry' if retry else ''), 'condition_id': condition, 'reference_id': reference,
            'repetition': 0, 'is_final_attempt': not retry, 'parent_final_result_id': ident if retry else None,
            'requested_model': 'openai/gpt-6.1-sol', 'reasoning_effort': condition,
            'harbor_job_id': 'job', 'reference_set': 'nv100-v1', 'cost_kind': 'test',
            'formal_fidelity_percent': score, 'visual_fidelity_percent': score, 'kinematic_fidelity_percent': score,
            'inference_cost_usd': cost, 'elapsed_seconds': 60, 'total_tokens': 10}


def test_retry_cost_uses_intended_reconstructions_and_does_not_change_final_scores():
    data = module.build([row('low','a',90,2),row('low','b',100,4),row('low','a',0,3,retry=True)], evaluator='2.2.0',revision='test')
    r = data['runs'][0]
    assert r['metrics']['cost_usd_per_asset']['value'] == 4.5
    assert r['metrics']['elapsed_seconds_per_asset']['value'] == 90
    assert r['scores']['formal']['value'] == 95


def test_matched_scores_use_same_slots_without_zero_filling():
    rows = [row('low','a',10,1),row('low','b',None,1),row('high','a',20,2),row('high','b',100,2)]
    data = module.build(rows,evaluator='2.2.0',revision='test')
    assert [r['metrics']['cost_usd_per_asset']['paired_scores']['formal']['value'] for r in data['runs']] == [10,20]
    assert [r['scores']['formal']['value'] for r in data['runs']] == [10,60]
    assert len({r['metrics']['cost_usd_per_asset']['paired_scores']['formal']['cohort_sha256'] for r in data['runs']}) == 1


def test_frontier_ties_and_strict_dominance_are_order_independent():
    points=[{'id':'a','x':1,'y':50},{'id':'tie','x':1,'y':50},{'id':'better','x':2,'y':90},{'id':'bad','x':3,'y':90},{'id':'null','x':None,'y':100}]
    assert {p['id'] for p in module.frontier(points)} == {'a','tie','better'}
    assert module.frontier(points) == module.frontier(list(reversed(points))) or {p['id'] for p in module.frontier(points)} == {p['id'] for p in module.frontier(list(reversed(points)))}


def test_unknown_usage_remains_counted_as_unknown():
    r=module.build([row('low','a',0,None)],evaluator='2.2.0',revision='test')['runs'][0]
    assert r['scores']['formal']['value'] == 0
    assert r['metrics']['cost_usd_per_asset']['value'] is None
    assert r['metrics']['cost_usd_per_asset']['unknown_execution_count'] == 1
