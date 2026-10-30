#!/usr/bin/env python3
"""Build reusable benchmark figures and chart data from published attempt rows."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
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
    counts = {len(trials) for trials in assets.values()}
    if len(counts) != 1 or min(counts) < k or any(set(trials) != set(range(len(trials))) for trials in assets.values()):
        raise ValueError('Every asset must have the same contiguous final repetitions, at least three')
    n = counts.pop()
    states = [v for trials in assets.values() for v in trials.values()]
    if any(v is not None and type(v) is not bool for v in states):
        raise ValueError('Formal success must be an explicit boolean or unknown')
    histogram = {str(c): 0 for c in range(n + 1)}
    for trials in assets.values():
        if all(v is not None for v in trials.values()):histogram[str(sum(trials.values()))] += 1
    def rate(bounds):
        lower = math.fsum(lo for lo, hi in bounds)
        upper = math.fsum(hi for lo, hi in bounds)
        unknown = sum(lo != hi for lo, hi in bounds)
        return {'value': 100 * lower / len(bounds) if not unknown else None,
                'successes': lower, 'denominator': len(bounds), 'unknown': unknown,
                'lower_bound': 100 * lower / len(bounds), 'upper_bound': 100 * upper / len(bounds)}
    def estimate(c, all_pass):
        return math.comb(c, k) / math.comb(n, k) if all_pass else 1 - math.comb(n-c, k) / math.comb(n, k)
    def asset_bounds(all_pass):
        result = []
        for trials in assets.values():
            c = sum(v is True for v in trials.values())
            unknown = sum(v is None for v in trials.values())
            result.append((estimate(c, all_pass), estimate(c+unknown, all_pass)))
        return result
    return {'criterion': 'all_46_mandatory_requirements_pass', 'k': k, 'samples_per_asset': n, 'assets':len(assets),
            'final_attempts':len(states), 'pass_at_1':rate([(int(v is True), int(v is not False)) for v in states]),
            'pass_at_3':rate(asset_bounds(False)), 'pass_all_3':rate(asset_bounds(True)), 'successes_per_asset':histogram,
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
    group_key = lambda r: (r['harbor_job_id'], r['condition_id'])
    for r in finals:
        groups[group_key(r)].append(r)
    for r in rows:
        parent = r['id'] if r['is_final_attempt'] else r['parent_final_result_id']
        assert parent in by_id and group_key(by_id[parent]) == group_key(r)
        attempts[group_key(r)].append(r)
    keys = lambda rs: {(r['reference_id'], r['repetition']): r for r in rs}
    indexed = {condition: keys(rs) for condition, rs in groups.items()}
    if any(len(indexed[c]) != len(rs) for c,rs in groups.items()):
        raise ValueError('Duplicate scored object/repetition slot')
    reference_sets = [{k[0] for k in rs} for rs in indexed.values()]
    if not all(s == reference_sets[0] for s in reference_sets):
        raise ValueError('Conditions must cover the same references')
    common = {metric: sorted(set.intersection(*[{k for k, r in rs.items()
        if known(r[f'{metric}_fidelity_percent'])} for rs in indexed.values()])) for metric in METRICS}
    runs = []
    for identity, rs in groups.items():
        first = rs[0]
        job, condition = identity
        identity_fields = ('requested_model', 'reasoning_effort', 'reference_set', 'harness', 'harness_id', 'authoring_label', 'agent_revision', 'agent_version')
        if any(tuple(r.get(k) for k in identity_fields) != tuple(first.get(k) for k in identity_fields) for r in rs):
            raise ValueError('Configuration metadata differs within a condition')
        family = 'Sol' if first['requested_model'].endswith('-sol') else 'Terra' if first['requested_model'].endswith('-terra') else first['requested_model']
        effort = first['reasoning_effort']
        display_effort = 'low' if family == 'Sol' and effort is None else effort or 'default'
        n = len(rs)
        configuration_id = first.get('configuration_id') or condition
        if sum(key[1] == condition for key in groups) > 1:
            configuration_id = f'{configuration_id}@{job}'
        harness = first.get('harness') or first.get('harness_id') or 'Codex'
        authoring = first.get('authoring_label') or harness
        agent_revision = first.get('agent_revision') or first.get('harness_version')
        if not isinstance(agent_revision, str) or len(agent_revision) != 40 or any(c not in '0123456789abcdef' for c in agent_revision):
            agent_revision = None
        model_name = {'Sol':'ChatGPT 6.1 Sol', 'Terra':'ChatGPT 6.1 Terra'}.get(family, family)
        run = {'id': configuration_id, 'label': f'{model_name} · {display_effort} · {authoring}', 'model': model_name, 'model_family': family,
               'model_display_name': model_name, 'reasoning_display': display_effort,
               'requested_model': first['requested_model'], 'reasoning': effort, 'harness': harness,
               'authoring_label': authoring, 'agent_revision': agent_revision,
               'evaluator_version': evaluator, 'job_id': job,
               'frontier_group': f'{first["reference_set"]}:{evaluator}',
               'reference_count': len({r['reference_id'] for r in rs}), 'final_slots': n,
               'execution_count': len(attempts[identity]),
               'source_url': f'https://huggingface.co/datasets/hcltech-robotics/simreadybench/tree/{revision}/results/harbor/{job}',
               'scores': {}, 'metrics': {}}
        for metric in METRICS:
            values = [r[f'{metric}_fidelity_percent'] for r in rs if known(r[f'{metric}_fidelity_percent'])]
            run['scores'][metric] = {'value': statistics.mean(values) if values else None,
                'measured_attempts': len(values), 'eligible_attempts': n, 'complete': len(values) == n,
                'status': 'known' if len(values) == n else 'partial' if values else 'not_applicable' if all(r.get(f'{metric}_fidelity_status') == 'not_applicable' for r in rs) else 'unknown'}
        for axis, (field, label, divisor) in AXES.items():
            values = [r[field] for r in attempts[identity] if known(r.get(field))]
            value = math.fsum(values) / n if values else None
            metric = {'value': value, 'unit': {'cost_usd_per_asset':'USD/asset','elapsed_seconds_per_asset':'seconds/asset','tokens_per_asset':'tokens/asset'}[axis], 'denominator': n,
                      'measured_attempts': len(values), 'total_attempts': len(attempts[identity]),
                      'unknown_execution_count': len(attempts[identity]) - len(values),
                      'sum_recorded': math.fsum(values) if values else None,
                      'complete': len(values) == len(attempts[identity]),
                      'includes_retries': True, 'paired_scores': {}}
            if field == 'inference_cost_usd':
                metric['kind'] = first['cost_kind']
            if field == 'elapsed_seconds':
                metric['timing_scope'] = 'sum of individual trial durations, including setup and verification; not job wall time'
            for score in METRICS:
                cohort = common[score]
                scores = [indexed[identity][key][f'{score}_fidelity_percent'] for key in cohort]
                metric['paired_scores'][score] = {'value': statistics.mean(scores) if scores else None,
                    'x_value': value, 'measured_attempts': len(scores), 'eligible_attempts': n,
                    'complete': len(scores) == n,
                    'cohort_sha256': hashlib.sha256(json.dumps(cohort, separators=(',', ':')).encode()).hexdigest()}
            run['metrics'][axis] = metric
        execution_rows = attempts[identity]
        run['tokens_per_asset'] = {name:resource_summary(execution_rows, field, n) for name,field in
            [('total','total_tokens'),('input','input_tokens'),('output','output_tokens'),('cached_input','cached_input_tokens')]}
        run['inference_cost_per_asset'] = resource_summary(execution_rows,'inference_cost_usd',n)
        run['trial_seconds_per_asset'] = resource_summary(execution_rows,'elapsed_seconds',n)
        run['cost_per_point'] = {metric:cost_per_point(run['inference_cost_per_asset'],run['scores'][metric]['value']) for metric in METRICS}
        run['pass_rates'] = pass_summary(rs,formal_success) if formal_success else None
        runs.append(run)
    order = {'low': 0, 'medium': 1, 'high': 2}
    runs.sort(key=lambda r: (r['model_family'], order.get(r['reasoning_display'], 3), r['label']))
    output = {'schema_version': 2, 'reference_set': finals[0]['reference_set'], 'reference_count': runs[0]['reference_count'],
              'evaluator_version': evaluator, 'source_revision': revision,
              'job_ids': sorted({r['harbor_job_id'] for r in finals}),
              'formal_fidelity_primary': True, 'runs': runs,
              'leaderboard_definitions': {'pass':'all 46 mandatory requirements pass',
                  'formal_success_source':'per-trial metrics.json: scores.formal.evidence.all_scored_mandatory_pass',
                  'pass_at_1':'mean asset success rate over final attempts',
                  'pass_at_3':'estimated probability of at least one formal pass in three attempts',
                  'pass_all_3':'estimated probability of three formal passes in three attempts',
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
            points = [{'id': r['id'], 'x': r['metrics'][axis]['value'] / AXES[axis][2] if known(r['metrics'][axis]['value']) else None,
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


def write_csv(data, output):
    fields = ['configuration', 'model', 'reasoning_effort_display', 'authoring', 'formal_percent', 'optional_criteria_percent', 'optional_lower_bound_percent', 'optional_upper_bound_percent', 'visual_percent',
              'kinematic_percent', 'pass_at_1_percent', 'pass_at_3_percent', 'pass_all_3_percent',
              'total_tokens_per_asset', 'input_tokens_per_asset', 'output_tokens_per_asset',
              'cached_input_tokens_per_asset', 'inference_usd_per_asset', 'inference_usd_per_formal_point',
              'trial_minutes_per_asset', 'resource_measurements_partial']
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in data['runs']:
            row = dict(zip(fields[:4], [run['label'], run['model_display_name'], run['reasoning_display'], run.get('authoring_label') or run['harness']]))
            row.update({f'{metric}_percent': run['scores'][metric]['value'] for metric in METRICS})
            optional = run.get('optional_criteria') or {}
            row.update(optional_criteria_percent=optional.get('value'), optional_lower_bound_percent=optional.get('lower_bound_percent'), optional_upper_bound_percent=optional.get('upper_bound_percent'))
            row.update({f'{name}_percent': (run['pass_rates'] or {}).get(name, {}).get('value') for name in ('pass_at_1', 'pass_at_3', 'pass_all_3')})
            row.update({f'{name}_tokens_per_asset': run['tokens_per_asset'][name]['value'] for name in ('total', 'input', 'output', 'cached_input')})
            row.update(inference_usd_per_asset=run['inference_cost_per_asset']['value'],
                       inference_usd_per_formal_point=run['cost_per_point']['formal']['value'],
                       trial_minutes_per_asset=run['trial_seconds_per_asset']['value']/60 if run['trial_seconds_per_asset']['value'] is not None else None,
                       resource_measurements_partial=any(metric['partial'] for metric in [run['inference_cost_per_asset'], run['trial_seconds_per_asset'], *run['tokens_per_asset'].values()]))
            writer.writerow(row)


def main():
    p = argparse.ArgumentParser();inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--attempts', type=Path, nargs='+');inputs.add_argument('--leaderboard',type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evaluator-version');p.add_argument('--source-revision');p.add_argument('--formal-evidence',type=Path,nargs='+');args=p.parse_args()
    if args.leaderboard:
        data=json.loads(args.leaderboard.read_text())
        if not data.get('formal_fidelity_primary') or not data.get('runs') or len({r['id'] for r in data['runs']}) != len(data['runs']):
            raise ValueError('Invalid published leaderboard')
    else:
        import pyarrow.parquet as pq
        if not args.evaluator_version or not args.source_revision:p.error('--attempts requires --evaluator-version and --source-revision')
        rows = [row for path in args.attempts for row in pq.read_table(path).to_pylist()]
        evidence = {}
        for path in args.formal_evidence or []:
            values = json.loads(path.read_text())
            if evidence.keys() & values.keys(): raise ValueError('Duplicate formal trial evidence')
            evidence.update(values)
        data=build(rows, evaluator=args.evaluator_version, revision=args.source_revision,formal_success=evidence)
    args.output.mkdir(parents=True, exist_ok=True);(args.output/'leaderboard.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');(args.output/'pareto.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');write_csv(data,args.output/'leaderboard.csv');figures(data,args.output)
    print(json.dumps({'configurations':len(data['runs']),'files':[p.name for p in args.output.iterdir()]}))

if __name__ == '__main__':main()
