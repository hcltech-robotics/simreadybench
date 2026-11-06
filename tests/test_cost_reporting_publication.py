import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('cost_publication', Path(__file__).parents[1] / 'scripts/correct_published_costs.py')
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run(identity, cost):
    final = {'value': cost, 'sum_recorded': cost * 300, 'denominator': 300,
             'known_executions': 300, 'unknown_executions': 0, 'partial': False, 'includes_retries': False}
    return {'id': identity, 'job_id': 'job', 'inference_cost_per_asset': final,
            'inference_spend': {'final': final, 'retry_overhead': {'sum_recorded': 2, 'unknown_executions': 1}}}


def test_summary_corrects_per_item_completeness_but_preserves_job_receipt():
    source = {'job_id': 'job', 'estimated_model_cost_usd': 123,
              'usage_all_executions_including_retries': {'inference_cost_usd': {'sum_recorded': 123}},
              'model_summaries': [{'id': 'cad', 'scores': {'formal': 100},
                                   'inference_cost_per_asset': {'partial': True}}]}
    revised = module.correct_summary(source, [run('cad', .4)])
    assert revised['model_summaries'][0]['inference_cost_per_asset']['partial'] is False
    assert revised['model_summaries'][0]['scores'] == {'formal': 100}
    assert revised['estimated_model_cost_usd'] == 123
    assert revised['usage_all_executions_including_retries']['inference_cost_usd']['sum_recorded'] == 123
    assert revised['final_output_inference_spend']['cad']['retry_overhead']['unknown_executions'] == 1


def test_comparison_separates_final_cost_change_from_recorded_spend_change():
    source = {'resources': {'baseline': {'cost_usd': 31}, 'cad': {'cost_usd': 28}},
              'resource_changes_percent': {'cost_usd': -9.68}, 'scores': {'formal': 100}}
    revised = module.correct_comparison(source, run('baseline', .1), run('cad', .08))
    assert revised['resource_changes_percent']['final_output_inference_cost_usd'] == pytest.approx(-20)
    assert revised['resource_changes_percent']['cost_usd'] == -9.68
    assert revised['resources']['baseline']['cost_usd'] == 31
    assert revised['resources']['baseline']['cost_usd_scope'] == 'retry_inclusive_recorded_spend'
    assert revised['scores'] == {'formal': 100}
