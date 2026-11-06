#!/usr/bin/env python3
"""Correct derived NV100 cost publications from a pinned, downloaded HF snapshot.

No source attempts or scores are written. Download the dataset's README, root
leaderboards, results/harbor/*/presentation/{leaderboard,pareto}.json and
viewer/harbor-v1/attempts/*.parquet at --revision into --source first.
Also supply published job-summary.json and comparison.{json,md} cost mirrors.
Review --output before publishing with the existing HF connection. The output
contains only derived reports, a card and source hashes, never model inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from build_nv100_results import correct_cost_reporting, write_csv
from plot_results import plot

NOTE = ('Inference cost uses final retained outputs, with one cost per final item. '
        'Retry inference spend is reported separately in JSON and CSV, including unknown retry costs. '
        'Tokens and trial time retain their retry-inclusive scope. '
        'Complete cost coverage describes measurement availability, not invoice verification: '
        'original Codex amounts are Harbor-reported with an unverified basis; CAD Codex amounts are '
        'estimated under openai-standard-global-30m and its retained assumptions. '
        'Other configurations retain their recorded cost kinds.')


def update_card(text, runs):
    lookup = {(r['model_display_name'], r['reasoning_display'], r.get('authoring_label') or r['harness']): r for r in runs}
    lines = text.splitlines(keepends=True)
    updated = 0
    for i, line in enumerate(lines):
        cells = [v.strip() for v in line.strip().split('|')[1:-1]]
        if len(cells) != 14 or tuple(cells[:3]) not in lookup:
            continue
        run = lookup[tuple(cells[:3])]
        cost, ratio = run['inference_cost_per_asset']['value'], run['cost_per_point']['formal']['value']
        cells[11] = f'${cost:.4f}' if cost is not None else '—'
        cells[12] = f'${ratio:.6f}' if ratio is not None else '—'
        lines[i] = '| ' + ' | '.join(cells) + ' |\n'
        updated += 1
    if updated != len(runs):
        raise ValueError('Card does not contain the exact published configuration table')
    text = ''.join(lines)
    text = text.replace('Cost, tokens and trial time include retries.', NOTE)
    text = text.replace('Inference cost, tokens and trial time include retries. Hosted compute is separate.',
                        'Inference cost uses final retained outputs. Tokens and trial time include retries. Retry spend and hosted compute are separate.')
    if NOTE not in text:
        marker = '\n[Full results and coverage]'
        if marker not in text:
            raise ValueError('No cost methodology insertion point')
        text = text.replace(marker, '\n' + NOTE + '\n' + marker)
    return text


def correct_summary(data, runs):
    """Add final-item accounting while preserving historical spend receipts."""
    job_runs = [r for r in runs if r['job_id'] == data['job_id']]
    data['final_output_inference_spend'] = {r['id']: r['inference_spend'] for r in job_runs}
    for summary in data.get('model_summaries', []):
        run = next(r for r in job_runs if r['id'] == summary['id'])
        summary['inference_cost_per_asset'] = run['inference_cost_per_asset']
        summary['inference_spend'] = run['inference_spend']
    data['cost_reporting_scope'] = 'Per-item inference costs use final retained outputs; existing job totals remain retry-inclusive recorded spend.'
    return data


def correct_comparison(data, baseline, cad):
    for key, run in [('baseline', baseline), ('cad', cad)]:
        resource = data['resources'][key]
        resource['cost_usd_scope'] = 'retry_inclusive_recorded_spend'
        resource['final_output_inference_cost'] = run['inference_cost_per_asset']
        resource['inference_spend'] = run['inference_spend']
    before = baseline['inference_cost_per_asset']['value']
    after = cad['inference_cost_per_asset']['value']
    data['resource_changes_percent']['final_output_inference_cost_usd'] = 100 * (after / before - 1)
    return data


def prepare(source, output, revision):
    rows, inputs = [], {}
    # Only read evaluator-2.2.0 jobs: older evaluator publications are historical.
    latest_path = Path('results/harbor/bb865915-d3cc-400f-bda1-0eaf4b3e4b4a/presentation/leaderboard.json')
    latest = json.loads((source / latest_path).read_text())
    for job in latest['job_ids']:
        path = Path(f'viewer/harbor-v1/attempts/{job}.parquet')
        raw = (source/path).read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        rows.extend(pq.read_table(source/path).to_pylist())
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.rglob('*.json')):
        if path.name not in ('leaderboard.json','pareto.json'):
            continue
        data = json.loads(path.read_text())
        if not data.get('runs', data.get('entries')) or data.get('evaluator_version') != '2.2.0':
            continue
        original = copy.deepcopy(data)
        correct_cost_reporting(data, rows)
        for before, after in zip(original.get('runs',original.get('entries')),data.get('runs',data.get('entries'))):
            assert before['scores'] == after['scores']
            for field in ('pass_rates','optional_criteria','tokens_per_asset','trial_seconds_per_asset'):
                assert before.get(field) == after.get(field)
        data['cost_source_revision'] = revision
        target = output/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        count += 1
        if path.name == 'leaderboard.json' and 'runs' in data:
            write_csv(data,target.with_suffix('.csv'))
            # Regenerate only the cost figures; other plots retain their data and bytes.
            work = output/'.figure-work'
            work.mkdir(exist_ok=True)
            plot(data,work,selected_axes={'cost_usd_per_asset'})
            for ext in ('svg','png'):
                (target.parent/f'pareto-cost.{ext}').write_bytes((work/f'pareto-cost.{ext}').read_bytes())
    latest = json.loads((output/latest_path).read_text())
    (output/'README.md').write_text(update_card((source/'README.md').read_text(), latest['runs']))

    for path in source.glob('results/harbor/*/job-summary.json'):
        data = json.loads(path.read_text())
        if data.get('job_id') not in latest['job_ids']:
            continue
        target = output/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(correct_summary(data, latest['runs']),indent=2,allow_nan=False)+'\n')
    comparison_path = Path('results/harbor/f5f94694-fb90-4593-9d21-8c5b1cb75e42/comparison.json')
    if (source/comparison_path).exists():
        baseline = next(r for r in latest['runs'] if r['job_id'] == '0f2b2409-7945-44b8-abb4-c4faef268035')
        cad = next(r for r in latest['runs'] if r['job_id'] == 'f5f94694-fb90-4593-9d21-8c5b1cb75e42')
        data = correct_comparison(json.loads((source/comparison_path).read_text()),baseline,cad)
        (output/comparison_path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        text = (source/comparison_path.with_suffix('.md')).read_text()
        headline = '| Final-output inference cost / item | $%.8f (300/300 costed) | $%.8f (300/300 costed) |\n' % (baseline['inference_cost_per_asset']['value'],cad['inference_cost_per_asset']['value'])
        headline += '| Final-output inference cost, total | $%.9f | $%.9f |\n' % (baseline['inference_cost_per_asset']['sum_recorded'],cad['inference_cost_per_asset']['sum_recorded'])
        text = text.replace('| Reported inference cost |', headline+'| Reported inference cost, retry-inclusive spend |')
        text = text.replace('reported inference cost falls 7.6%.', 'final-output inference cost falls %.1f%%; retry-inclusive recorded inference spend falls 7.6%%.' % -data['resource_changes_percent']['final_output_inference_cost_usd'])
        text = text.replace('resource sums include every retry.', 'token, time and recorded-spend sums include every retry. Final-output inference pricing excludes retries; separate retry accounting is retained in comparison.json.')
        (output/comparison_path.with_suffix('.md')).write_text(text)

    report = {'version':'final-retained-outputs-v1','source_revision':revision,'attempt_inputs_sha256':inputs,
              'configurations':len(latest['runs']),'final_items':sum(r['final_slots'] for r in latest['runs']),
              'final_cost_known_items':sum(r['inference_cost_per_asset']['known_executions'] for r in latest['runs']),
              'retry_attempts':sum(r['execution_count']-r['final_slots'] for r in latest['runs']),
              'retry_cost_unknown_attempts':sum(r['inference_spend']['retry_overhead']['unknown_executions'] for r in latest['runs']),
              'corrected_json_files':count,
              'additional_cost_summary_mirrors':len(list(output.glob('results/harbor/*/job-summary.json'))) + (2 if (output/comparison_path).exists() else 0),
              'scores_and_attempts_changed':False}
    (output/'results/cost-reporting-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--revision',required=True);a=p.parse_args()
    if len(a.revision)!=40 or any(c not in '0123456789abcdef' for c in a.revision):p.error('A full pinned revision is required')
    print(json.dumps(prepare(a.source,a.output,a.revision),indent=2))
