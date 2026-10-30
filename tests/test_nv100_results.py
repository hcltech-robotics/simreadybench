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


def three_rows(condition='low', scores=(100,100,100)):
    result=[]
    for repetition,score in enumerate(scores):
        r=row(condition,'asset',score,1)
        r['id']=condition+str(repetition)
        r['harbor_trial_id']=r['id']
        r['repetition']=repetition
        result.append(r)
    return result


def test_pass_rates_are_assetwise_at_least_one_and_all_three_not_cubed_mean():
    rows=[];evidence={}
    for count in range(4):
        for repetition,r in enumerate(three_rows()):
            r['reference_id']=f'asset-{count}';r['harbor_trial_id']=f'{count}-{repetition}'
            rows.append(r);evidence[r['harbor_trial_id']]={'success':repetition<count}
    values=module.pass_summary(rows,evidence)
    assert values['pass_at_1']['value']==50
    assert values['pass_at_3']['value']==75
    assert values['pass_all_3']['value']==25
    assert values['successes_per_asset']=={'0':1,'1':1,'2':1,'3':1}


def test_pass_rates_reject_duplicate_and_incomplete_scored_repetitions():
    rows=three_rows();evidence={r['harbor_trial_id']:{'success':True} for r in rows}
    with pytest.raises(ValueError,match='three'):
        module.pass_summary(rows[:2],evidence)
    with pytest.raises(ValueError,match='Duplicate'):
        module.pass_summary(rows+[rows[0]],evidence)


def test_missing_binary_flag_stays_unknown_without_becoming_a_failure():
    rows=three_rows();evidence={rows[0]['harbor_trial_id']:{'success':True},rows[2]['harbor_trial_id']:{'success':False}}
    values=module.pass_summary(rows,evidence)
    assert values['pass_at_1']['value'] is None
    assert values['pass_at_3']['value']==100
    assert values['pass_all_3']['value']==0
    assert module.formal_pass_from_report({'scores':{'formal':{'score_percent':100}}}) is None


def test_explicit_mandatory_flag_is_authoritative_and_complete_profile_required():
    evidence={'all_scored_mandatory_pass':False,'applicable_count':46,'profile_mandatory_count':46,'unknown_count':0}
    assert module.formal_pass_from_report({'scores':{'formal':{'score_percent':100,'evidence':evidence}}}) is False
    evidence={**evidence,'all_scored_mandatory_pass':True}
    assert module.formal_pass_from_report({'formal_report':{'simready':evidence}}) is True
    with pytest.raises(ValueError,match='46'):
        module.formal_pass_from_report({'formal_report':{'simready':{**evidence,'applicable_count':45}}})


def test_retry_counts_do_not_add_pass_samples_and_cached_input_is_not_added_twice():
    rows=three_rows(scores=(50,50,50))
    for r in rows:r.update(input_tokens=100,output_tokens=20,cached_input_tokens=80,total_tokens=120)
    evidence={r['harbor_trial_id']:{'success':False} for r in rows}
    retry={**rows[0],'id':'retry','is_final_attempt':False,'parent_final_result_id':rows[0]['id']}
    data=module.build(rows+[retry],evaluator='2.2.0',revision='test',formal_success=evidence)['runs'][0]
    assert data['pass_rates']['final_attempts']==3
    assert data['pass_rates']['pass_at_3']['value']==0
    assert data['tokens_per_asset']['total']['value']==160
    assert data['tokens_per_asset']['input']['value']+data['tokens_per_asset']['output']['value']==160
    assert data['tokens_per_asset']['cached_input']['value']==pytest.approx(320/3)


def test_cost_per_point_uses_ratio_of_condition_means_and_zero_rules():
    rows=three_rows(scores=(50,100,100))
    for r,cost in zip(rows,[2,8,20]):r['inference_cost_usd']=cost
    run=module.build(rows,evaluator='2.2.0',revision='test')['runs'][0]
    assert run['cost_per_point']['formal']['value']==pytest.approx(0.12)
    assert module.cost_per_point({'value':1},0)['status']=='infinite'
    assert module.cost_per_point({'value':0},0)['status']=='undefined'
    assert module.cost_per_point({'value':None},100)['status']=='unknown'
    assert module.cost_per_point({'value':0},100)['value']==0
    assert module.cost_per_point({'value':1,'partial':True},100)['status']=='partial_lower_bound'


def test_time_storage_unit_is_seconds_while_display_uses_minutes_once():
    rows=[row('low','a',90,1),row('low','b',100,1)]
    rows[0]['elapsed_seconds']=120
    rows[1]['elapsed_seconds']=60
    run=module.build(rows,evaluator='2.2.0',revision='test')['runs'][0]
    metric=run['metrics']['elapsed_seconds_per_asset']
    assert metric['unit']=='seconds/asset'
    assert metric['value']==90
    assert metric['value']/module.AXES['elapsed_seconds_per_asset'][2]==1.5
    assert run['trial_seconds_per_asset']['value']==90


def test_incomplete_cost_frontier_is_explicitly_provisional():
    rows=[row('low','a',90,1),row('low','b',100,None)]
    run=module.build(rows,evaluator='2.2.0',revision='test')['runs'][0]
    assert run['metrics']['cost_usd_per_asset']['paired_scores']['formal']['frontier_provisional'] is True
    assert run['metrics']['elapsed_seconds_per_asset']['paired_scores']['formal']['frontier_provisional'] is False


def test_full_model_display_names_keep_raw_ids_and_reasoning_settings():
    sol=row('low','a',100,1);sol['reasoning_effort']=None
    terra=row('high','a',100,1);terra['requested_model']='openai/gpt-6.1-terra'
    runs=module.build([sol,terra],evaluator='2.2.0',revision='test')['runs']
    assert [r['model_display_name'] for r in runs]==['ChatGPT 6.1 Sol','ChatGPT 6.1 Terra']
    assert runs[0]['requested_model']=='openai/gpt-6.1-sol' and runs[0]['reasoning'] is None
    assert runs[0]['reasoning_display']=='low' and runs[1]['reasoning_display']=='high'
    assert runs[1]['requested_model']=='openai/gpt-6.1-terra'


def four_rows():
    rows = three_rows('medium')
    rows.append({**rows[-1], 'id':'medium3', 'harbor_trial_id':'medium3', 'repetition':3})
    return rows


def test_four_attempt_pass_at_three_averages_subsets_and_preserves_unknown_bounds():
    rows = four_rows()
    for successes, expected_any, expected_all in [(0,0,0),(1,75,0),(2,100,0),(3,100,25),(4,100,100)]:
        evidence = {r['harbor_trial_id']:{'success':i<successes} for i,r in enumerate(rows)}
        rates = module.pass_summary(rows,evidence)
        assert rates['samples_per_asset'] == 4
        assert rates['pass_at_3']['value'] == expected_any
        assert rates['pass_all_3']['value'] == expected_all
    evidence = {r['harbor_trial_id']:{'success':True} for r in rows[:3]}
    rates = module.pass_summary(rows,evidence)
    assert rates['pass_at_3']['value'] == 100
    assert rates['pass_all_3']['value'] is None
    assert rates['pass_all_3']['lower_bound'] == 25
    assert rates['pass_all_3']['upper_bound'] == 100


def test_three_and_four_attempt_configurations_keep_own_means_and_match_chart_slots():
    baseline = three_rows('medium',scores=(60,60,60))
    cad = four_rows()
    for r in cad:
        r.update(id='cad-'+r['id'],harbor_trial_id='cad-'+r['harbor_trial_id'],harbor_job_id='cad-job',
                 harness='Codex',authoring_label='Codex / build123d',agent_revision='cb68fc11')
    cad[-1]['formal_fidelity_percent'] = 0
    retry = {**cad[0], 'id':'retry', 'is_final_attempt':False, 'parent_final_result_id':cad[0]['id'], 'inference_cost_usd':None}
    evidence = {r['harbor_trial_id']:{'success':True} for r in baseline+cad}
    runs = module.build(baseline+cad+[retry],evaluator='2.2.0',revision='test',formal_success=evidence)['runs']
    assert len({r['id'] for r in runs}) == 2
    old, new = sorted(runs,key=lambda r:r['final_slots'])
    assert old['final_slots'] == 3 and new['final_slots'] == 4
    assert old['scores']['formal']['value'] == 60 and new['scores']['formal']['value'] == 75
    assert new['metrics']['cost_usd_per_asset']['paired_scores']['formal']['value'] == 100
    assert old['metrics']['cost_usd_per_asset']['paired_scores']['formal']['cohort_sha256'] == new['metrics']['cost_usd_per_asset']['paired_scores']['formal']['cohort_sha256']
    assert new['inference_cost_per_asset']['denominator'] == 4
    assert new['inference_cost_per_asset']['unknown_executions'] == 1
    assert new['execution_count'] == 5
    assert new['pass_rates']['final_attempts'] == 4
    assert new['authoring_label'] == 'Codex / build123d'


def test_csv_export_retains_nulls_and_separates_authoring_configurations(tmp_path):
    import csv
    runs = module.build([row('medium','asset',None,None)],evaluator='2.2.0',revision='test')
    module.write_csv(runs,tmp_path/'leaderboard.csv')
    record = next(csv.DictReader((tmp_path/'leaderboard.csv').open()))
    assert record['formal_percent'] == '' and record['inference_usd_per_asset'] == ''
    assert record['model'] == 'ChatGPT 6.1 Sol' and record['authoring'] == 'Codex'
