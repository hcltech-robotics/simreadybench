#!/usr/bin/env python3
"""Build reusable benchmark figures and chart data from published attempt rows."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

METRICS = ('formal', 'visual', 'kinematic')
AXES = {'cost_usd_per_asset': ('inference_cost_usd', 'Inference cost ($/asset)', 1),
        'elapsed_seconds_per_asset': ('elapsed_seconds', 'Trial time (min/asset)', 60),
        'tokens_per_asset': ('total_tokens', 'Tokens/asset', 1)}
COLORS = ['#1677aa', '#40a9b6', '#154f78', '#df8a37', '#a74d24']


def known(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def frontier(points):
    """Minimise resource use and maximise score; ties do not dominate."""
    eligible = [p for p in points if known(p['x']) and known(p['y'])]
    return sorted([p for p in eligible if not any(q['x'] <= p['x'] and q['y'] >= p['y']
        and (q['x'] < p['x'] or q['y'] > p['y']) for q in eligible)], key=lambda p: (p['x'], -p['y']))


def formal_pass_from_report(report):
    """Read the explicit mandatory-pass outcome; never infer it from a grade."""
    candidates = [(report.get('scores', {}).get('formal', {}) or {}).get('evidence', {}),
                  (report.get('formal_report', {}) or {}).get('simready', {})]
    for evidence in candidates:
        if not isinstance(evidence, dict):
            continue
        if type(evidence.get('all_scored_mandatory_pass')) is not bool:
            continue
        if evidence.get('applicable_count') != 46 or evidence.get('profile_mandatory_count') != 46:
            raise ValueError('Formal pass requires the complete 46-requirement profile')
        if evidence.get('unknown_count'):
            return None
        return evidence['all_scored_mandatory_pass']
    return None


def pass_summary(final_rows, formal_success, *, k=3):
    """Asset-wise combinatorial pass@k and pass^k, excluding retry executions."""
    assets = defaultdict(dict)
    for row in final_rows:
        trials = assets[row['reference_id']]
        repetition = row['repetition']
        if repetition in trials:
            raise ValueError('Duplicate scored object/repetition slot')
        trials[repetition] = formal_success.get(row['harbor_trial_id'], {}).get('success')
    if any(set(trials) != set(range(k)) for trials in assets.values()):
        raise ValueError('Every asset must have exactly three distinct final repetitions')
    states = [v for trials in assets.values() for v in trials.values()]
    if any(v is not None and type(v) is not bool for v in states):
        raise ValueError('Formal success must be an explicit boolean or unknown')
    known_states = [v for v in states if v is not None]
    histogram = {str(c): 0 for c in range(k + 1)}
    for trials in assets.values():
        if all(v is not None for v in trials.values()):histogram[str(sum(trials.values()))] += 1
    any_pass = [True if any(v is True for v in t.values()) else False if all(v is False for v in t.values()) else None for t in assets.values()]
    all_pass = [False if any(v is False for v in t.values()) else True if all(v is True for v in t.values()) else None for t in assets.values()]
    def rate(outcomes):
        measured = [v for v in outcomes if v is not None]
        successes = sum(measured)
        return {'value': 100 * successes / len(outcomes) if len(measured) == len(outcomes) else None,
                'successes': successes, 'denominator': len(outcomes), 'unknown': len(outcomes)-len(measured),
                'lower_bound': 100 * successes / len(outcomes),
                'upper_bound': 100 * (successes+len(outcomes)-len(measured)) / len(outcomes)}
    return {'criterion': 'all_46_mandatory_requirements_pass', 'k': k, 'assets':len(assets),
            'final_attempts':len(states), 'pass_at_1':rate(states), 'pass_at_3':rate(any_pass),
            'pass_all_3':rate(all_pass), 'successes_per_asset':histogram,
            'estimators': {'pass_at_k':'1 - C(n-c,k)/C(n,k)', 'pass_all_k':'C(c,k)/C(n,k)'},
            'retries_excluded':True,
            'by_asset':[{'reference_id':asset,'samples':len(trials),'successes':sum(v is True for v in trials.values()),
                         'unknown':sum(v is None for v in trials.values())} for asset,trials in sorted(assets.items())]}


def resource_summary(executions, field, intended):
    values = [r[field] for r in executions if known(r.get(field))]
    return {'value':math.fsum(values)/intended if values else None,
            'sum_recorded':math.fsum(values) if values else None, 'denominator':intended,
            'known_executions':len(values), 'unknown_executions':len(executions)-len(values),
            'partial':len(values)!=len(executions), 'includes_retries':True}


def cost_per_point(cost, score):
    if not known(cost.get('value')) or not known(score):return {'value':None,'status':'unknown'}
    if score == 0:return {'value':None,'status':'infinite' if cost['value'] > 0 else 'undefined'}
    return {'value':cost['value']/score,'status':'partial_lower_bound' if cost.get('partial') else 'known',
            'unit':'USD per fidelity point on a 0–100 scale'}


def build(rows, *, evaluator, revision, formal_success=None):
    formal_success = formal_success or {}
    for row in rows:
        if known(row.get('input_tokens')) and known(row.get('output_tokens')) and known(row.get('total_tokens')):
            if row['input_tokens'] + row['output_tokens'] != row['total_tokens']:
                raise ValueError('Total tokens must equal input plus output, excluding cached-input double counting')
        if known(row.get('cached_input_tokens')) and known(row.get('input_tokens')) and row['cached_input_tokens'] > row['input_tokens']:
            raise ValueError('Cached input must be a subset of input tokens')
    finals = [r for r in rows if r['is_final_attempt']]
    by_id = {r['id']: r for r in finals}
    assert len(by_id) == len(finals)
    groups = defaultdict(list)
    attempts = defaultdict(list)
    for r in finals:
        groups[r['condition_id']].append(r)
    for r in rows:
        parent = r['id'] if r['is_final_attempt'] else r['parent_final_result_id']
        assert parent in by_id and by_id[parent]['condition_id'] == r['condition_id']
        attempts[r['condition_id']].append(r)
    keys = lambda rs: {(r['reference_id'], r['repetition']): r for r in rs}
    indexed = {condition: keys(rs) for condition, rs in groups.items()}
    if any(len(indexed[c]) != len(rs) for c,rs in groups.items()):
        raise ValueError('Duplicate scored object/repetition slot')
    all_keys = [set(rs) for rs in indexed.values()]
    if not all(k == all_keys[0] for k in all_keys):
        raise ValueError('Conditions must cover the same reference/repetition slots')
    common = {metric: sorted(set.intersection(*[{k for k, r in rs.items()
        if known(r[f'{metric}_fidelity_percent'])} for rs in indexed.values()])) for metric in METRICS}
    runs = []
    for condition, rs in groups.items():
        first = rs[0]
        family = 'Sol' if first['requested_model'].endswith('-sol') else 'Terra' if first['requested_model'].endswith('-terra') else first['requested_model']
        effort = first['reasoning_effort']
        display_effort = 'low' if family == 'Sol' and effort is None else effort or 'default'
        n = len(rs)
        job = first['harbor_job_id']
        model_name = {'Sol':'ChatGPT 6.1 Sol', 'Terra':'ChatGPT 6.1 Terra'}.get(family, family)
        run = {'id': condition, 'label': f'{model_name} · {display_effort}', 'model': model_name, 'model_family': family,
               'model_display_name': model_name, 'reasoning_display': display_effort,
               'requested_model': first['requested_model'], 'reasoning': effort, 'harness': 'Codex',
               'evaluator_version': evaluator, 'job_id': job,
               'frontier_group': f'{first["reference_set"]}:{evaluator}',
               'reference_count': len({r['reference_id'] for r in rs}), 'final_slots': n,
               'execution_count': len(attempts[condition]),
               'source_url': f'https://huggingface.co/datasets/hcltech-robotics/simreadybench/tree/{revision}/results/harbor/{job}',
               'scores': {}, 'metrics': {}}
        for metric in METRICS:
            values = [r[f'{metric}_fidelity_percent'] for r in rs if known(r[f'{metric}_fidelity_percent'])]
            run['scores'][metric] = {'value': statistics.mean(values) if values else None,
                'measured_attempts': len(values), 'eligible_attempts': n, 'complete': len(values) == n,
                'status': 'known' if len(values) == n else 'partial' if values else 'not_applicable' if all(r.get(f'{metric}_fidelity_status') == 'not_applicable' for r in rs) else 'unknown'}
        for axis, (field, label, divisor) in AXES.items():
            values = [r[field] for r in attempts[condition] if known(r[field])]
            value = math.fsum(values) / n if values else None
            metric = {'value': value, 'unit': {'cost_usd_per_asset':'USD/asset','elapsed_seconds_per_asset':'seconds/asset','tokens_per_asset':'tokens/asset'}[axis], 'denominator': n,
                      'measured_attempts': len(values), 'total_attempts': len(attempts[condition]),
                      'unknown_execution_count': len(attempts[condition]) - len(values),
                      'sum_recorded': math.fsum(values) if values else None,
                      'complete': len(values) == len(attempts[condition]),
                      'includes_retries': True, 'paired_scores': {}}
            if field == 'inference_cost_usd':
                metric['kind'] = first['cost_kind']
            if field == 'elapsed_seconds':
                metric['timing_scope'] = 'sum of individual trial durations, including setup and verification; not job wall time'
            for score in METRICS:
                cohort = common[score]
                scores = [indexed[condition][key][f'{score}_fidelity_percent'] for key in cohort]
                metric['paired_scores'][score] = {'value': statistics.mean(scores) if scores else None,
                    'x_value': value, 'measured_attempts': len(scores), 'eligible_attempts': n,
                    'complete': len(scores) == n,
                    'cohort_sha256': hashlib.sha256(json.dumps(cohort, separators=(',', ':')).encode()).hexdigest()}
            run['metrics'][axis] = metric
        execution_rows = attempts[condition]
        run['tokens_per_asset'] = {name:resource_summary(execution_rows, field, n) for name,field in
            [('total','total_tokens'),('input','input_tokens'),('output','output_tokens'),('cached_input','cached_input_tokens')]}
        run['inference_cost_per_asset'] = resource_summary(execution_rows,'inference_cost_usd',n)
        run['trial_seconds_per_asset'] = resource_summary(execution_rows,'elapsed_seconds',n)
        run['cost_per_point'] = {metric:cost_per_point(run['inference_cost_per_asset'],run['scores'][metric]['value']) for metric in METRICS}
        run['pass_rates'] = pass_summary(rs,formal_success) if formal_success else None
        runs.append(run)
    order = {'low': 0, 'medium': 1, 'high': 2}
    runs.sort(key=lambda r: (r['model_family'], order.get(r['label'].split()[-1], 3), r['label']))
    output = {'schema_version': 2, 'reference_set': finals[0]['reference_set'], 'reference_count': runs[0]['reference_count'],
              'evaluator_version': evaluator, 'source_revision': revision,
              'job_ids': sorted({r['harbor_job_id'] for r in finals}),
              'formal_fidelity_primary': True, 'runs': runs,
              'leaderboard_definitions': {'pass':'all 46 mandatory requirements pass',
                  'formal_success_source':'per-trial metrics.json: scores.formal.evidence.all_scored_mandatory_pass',
                  'pass_at_1':'mean asset success rate over the three final attempts',
                  'pass_at_3':'at least one formal pass among three final attempts per asset',
                  'pass_all_3':'all three final attempts pass for an asset',
                  'cost_per_point':'inference cost per intended reconstruction divided by mean fidelity on the 0–100 scale',
                  'tokens':'total = input + output; cached input is already part of input',
                  'resources':'retry-inclusive recorded sums divided by intended reconstruction slots'},
              'methodology': 'Scores use the same object/repetition pairs across configurations.',
              'aggregation': {'scores': 'means over available final scores', 'pareto_scores': 'common measured object/repetition slots across configurations, separately per fidelity metric',
                  'resources': 'sum available execution measurements including retries, divided by intended reconstruction slots',
                  'missing': 'null; never imputed as zero', 'hosted_compute': 'job-level only; not allocated to configurations'}}
    for axis in AXES:
        for metric in METRICS:
            points = [{'id': r['id'], 'x': r['metrics'][axis]['value'],
                       'y': r['metrics'][axis]['paired_scores'][metric]['value']} for r in runs]
            ids = {p['id'] for p in frontier(points)}
            for run in runs:
                run['metrics'][axis]['paired_scores'][metric]['frontier'] = run['id'] in ids
                run['metrics'][axis]['paired_scores'][metric]['frontier_provisional'] = not run['metrics'][axis]['complete']
    return output


def figures(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    runs = data['runs']; labels = [r['label'] for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.1, len(runs) * .48 + 1.4)), layout='constrained')
    for ax, metric in zip(axes, METRICS):
        values = [r['scores'][metric]['value'] for r in runs]
        ax.barh(np.arange(len(runs)), [v if v is not None else np.nan for v in values], color=[COLORS[i % len(COLORS)] for i in range(len(runs))], height=.58)
        ax.set_yticks(range(len(runs)), labels);ax.invert_yaxis();ax.set_xlim(0, 113)
        ax.set_xticks([0, 25, 50, 75, 100]);ax.set_xlabel('Fidelity (%)');ax.set_title(metric.capitalize() + ' fidelity', fontweight='bold' if metric == 'formal' else 'normal')
        ax.grid(axis='x', alpha=.12);ax.set_axisbelow(True)
        for i, v in enumerate(values):
            if v is not None: ax.text(v + 1.5, i, f'{v:.2f}', va='center', fontsize=9)
    for ext in ('png', 'svg'): fig.savefig(output/f'scores.{ext}', dpi=180)
    plt.close(fig)
    for axis in ('cost_usd_per_asset', 'elapsed_seconds_per_asset'):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
        for ax, metric in zip(axes, METRICS):
            points = [{'id': r['id'], 'x': r['metrics'][axis]['value'] / AXES[axis][2],
                       'y': r['metrics'][axis]['paired_scores'][metric]['value']} for r in runs]
            fs = frontier(points)
            provisional_line = any(not r['metrics'][axis]['complete'] and r['id'] in {p['id'] for p in fs} for r in runs)
            if fs: ax.step([p['x'] for p in fs], [p['y'] for p in fs], where='post', color='#64748b', linewidth=1.6, linestyle='--' if provisional_line else '-', zorder=1)
            for i, (r, point) in enumerate(zip(runs, points)):
                if not known(point['x']) or not known(point['y']): continue
                ax.scatter(point['x'], point['y'], facecolors='white' if not r['metrics'][axis]['complete'] else COLORS[i % len(COLORS)], edgecolors=COLORS[i % len(COLORS)], linewidths=1.6, s=70 if r['metrics'][axis]['paired_scores'][metric]['frontier'] else 45, zorder=3)
                offset = [(5, 7), (5, -13), (-4, 8), (5, -12), (5, 7)][i % 5]
                if axis == 'elapsed_seconds_per_asset' and metric == 'kinematic':
                    if r['model_family'] == 'Terra' and r['reasoning_display'] == 'high': offset = (8, -17)
                    elif r['model_family'] == 'Sol' and r['reasoning_display'] == 'medium': offset = (7, 9)
                ax.annotate(r['label'], (point['x'], point['y']), xytext=offset, textcoords='offset points', fontsize=8,
                            ha='right' if offset[0] < 0 else 'left')
            ax.set_xlabel(AXES[axis][1]);ax.set_ylabel(metric.capitalize() + ' fidelity (%)')
            ax.set_title(metric.capitalize(), fontweight='bold' if metric == 'formal' else 'normal');ax.grid(alpha=.15);ax.margins(x=.25,y=.3)
        fig.suptitle('Quality–cost trade-offs' if axis.startswith('cost') else 'Quality–time trade-offs', fontsize=15, fontweight='bold')
        if any(not r['metrics'][axis]['complete'] for r in runs):
            from matplotlib.lines import Line2D
            fig.legend(handles=[Line2D([],[],marker='o',markerfacecolor='white',markeredgecolor='#475569',linestyle='none',label='Cost lower bound'),Line2D([],[],color='#64748b',linestyle='--',label='Provisional frontier')],loc='outside lower center',ncol=2,frameon=False,fontsize=9)
        name = 'pareto-cost' if axis.startswith('cost') else 'pareto-time'
        for ext in ('png', 'svg'):fig.savefig(output/f'{name}.{ext}', dpi=180)
        plt.close(fig)


def main():
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser();p.add_argument('--attempts', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evaluator-version', required=True);p.add_argument('--source-revision', required=True);p.add_argument('--formal-evidence',type=Path);args=p.parse_args()
    data=build(pq.read_table(args.attempts).to_pylist(), evaluator=args.evaluator_version, revision=args.source_revision,formal_success=json.loads(args.formal_evidence.read_text()) if args.formal_evidence else None)
    args.output.mkdir(parents=True, exist_ok=True);(args.output/'leaderboard.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');(args.output/'pareto.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');figures(data,args.output)
    print(json.dumps({'configurations':len(data['runs']),'files':[p.name for p in args.output.iterdir()]}))

if __name__ == '__main__':main()
