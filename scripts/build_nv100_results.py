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


def build(rows, *, evaluator, revision):
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
        run = {'id': condition, 'label': f'{family} {display_effort}', 'model': family, 'model_family': family,
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
                'measured_attempts': len(values), 'eligible_attempts': n, 'complete': len(values) == n}
        for axis, (field, label, divisor) in AXES.items():
            values = [r[field] for r in attempts[condition] if known(r[field])]
            value = math.fsum(values) / n if values else None
            metric = {'value': value, 'unit': label, 'denominator': n,
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
        runs.append(run)
    order = {'low': 0, 'medium': 1, 'high': 2}
    runs.sort(key=lambda r: (r['model_family'], order.get(r['label'].split()[-1], 3), r['label']))
    output = {'schema_version': 2, 'reference_set': finals[0]['reference_set'], 'reference_count': runs[0]['reference_count'],
              'evaluator_version': evaluator, 'source_revision': revision,
              'job_ids': sorted({r['harbor_job_id'] for r in finals}),
              'formal_fidelity_primary': True, 'runs': runs,
              'methodology': 'Each Pareto panel uses the same object/repetition pairs across configurations.',
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
            if fs: ax.step([p['x'] for p in fs], [p['y'] for p in fs], where='post', color='#64748b', linewidth=1.6, zorder=1)
            for i, (r, point) in enumerate(zip(runs, points)):
                if not known(point['x']) or not known(point['y']): continue
                ax.scatter(point['x'], point['y'], color=COLORS[i % len(COLORS)], s=70 if r['metrics'][axis]['paired_scores'][metric]['frontier'] else 45, zorder=3)
                offset = [(5, 7), (5, -13), (-4, 8), (5, -12), (5, 7)][i % 5]
                if axis == 'elapsed_seconds_per_asset' and metric == 'kinematic':
                    if r['label'] == 'Terra high': offset = (8, -17)
                    elif r['label'] == 'Sol medium': offset = (7, 9)
                ax.annotate(r['label'], (point['x'], point['y']), xytext=offset, textcoords='offset points', fontsize=8,
                            ha='right' if offset[0] < 0 else 'left')
            ax.set_xlabel(AXES[axis][1]);ax.set_ylabel(metric.capitalize() + ' fidelity (%)')
            ax.set_title(metric.capitalize(), fontweight='bold' if metric == 'formal' else 'normal');ax.grid(alpha=.15);ax.margins(x=.25,y=.3)
        fig.suptitle('Quality–cost trade-offs' if axis.startswith('cost') else 'Quality–time trade-offs', fontsize=15, fontweight='bold')
        name = 'pareto-cost' if axis.startswith('cost') else 'pareto-time'
        for ext in ('png', 'svg'):fig.savefig(output/f'{name}.{ext}', dpi=180)
        plt.close(fig)


def main():
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser();p.add_argument('--attempts', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evaluator-version', required=True);p.add_argument('--source-revision', required=True);args=p.parse_args()
    data=build(pq.read_table(args.attempts).to_pylist(), evaluator=args.evaluator_version, revision=args.source_revision)
    args.output.mkdir(parents=True, exist_ok=True);(args.output/'pareto.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');figures(data,args.output)
    print(json.dumps({'configurations':len(data['runs']),'files':[p.name for p in args.output.iterdir()]}))

if __name__ == '__main__':main()
