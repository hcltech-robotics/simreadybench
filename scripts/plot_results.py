#!/usr/bin/env python3
"""Render the published leaderboard with the project's model marker convention."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nv100_results import frontier, known


def style(run):
    model = (run['requested_model'] + ' ' + run['model']).lower()
    colour = '#76b900' if 'nemotron' in model else '#0072b2' if 'gpt-' in model else '#d97757' if 'claude' in model else '#7851a9'
    initial = 'N' if 'nemotron' in model else 'D' if 'deepseek' in model else 'C' if 'claude' in model else 'S' if 'sol' in model else 'T'
    custom = (run.get('harness_id') or run['harness']) not in ('codex', 'Codex', 'terminus-2', 'Terminus 2', 'claude-code', 'Claude Code')
    return colour, initial, custom


def plot(data, output, *, selected_axes=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none', 'axes.spines.top':False,'axes.spines.right':False})
    output.mkdir(parents=True, exist_ok=True)
    axes = [('cost_usd_per_asset', 1, 'Inference cost ($/asset)', 'cost'),
            ('elapsed_seconds_per_asset', 60, 'Trial time (min/asset)', 'time'),
            ('tokens_per_asset', 1e6, 'Tokens (millions/asset)', 'tokens')]
    runs = data['runs']
    for axis, divisor, label, suffix in axes:
        if selected_axes is not None and axis not in selected_axes: continue
        fig, panels = plt.subplots(1,3,figsize=(15,5.5),layout='constrained')
        for panel, metric in zip(panels, ['formal','visual','kinematic']):
            points=[{'id':r['id'],'x':r['metrics'][axis]['value'],'y':r['metrics'][axis]['paired_scores'][metric]['value']} for r in runs]
            fs=frontier(points);ids={p['id'] for p in fs}
            provisional = any(not r['metrics'][axis]['complete'] and r['id'] in ids for r in runs)
            if fs:panel.step([p['x']/divisor for p in fs],[p['y'] for p in fs],where='post',color='#64748b',linewidth=1.2,linestyle='--' if provisional else '-',zorder=1)
            for r,p in zip(runs,points):
                if not known(p['x']) or not known(p['y']):continue
                colour,initial,custom=style(r);x,y=p['x']/divisor,p['y'];filled=p['id'] in ids
                panel.scatter([x],[y],s=150,facecolors=colour if filled else 'white',edgecolors=colour,linewidths=1.5,zorder=3)
                if custom:panel.scatter([x],[y],s=245,facecolors='none',edgecolors=colour,linewidths=1,zorder=2)
                panel.text(x,y,initial,ha='center',va='center',fontsize=7,fontweight='bold',color='white' if filled else colour,zorder=4)
                effort=r['reasoning_display'];panel.annotate(effort,(x,y),xytext=(9,-12 if custom else 8),textcoords='offset points',fontsize=7,color='#344552')
            panel.set_title(metric.capitalize()+' fidelity',fontweight='bold' if metric=='formal' else 'normal');panel.set_xlabel(label);panel.set_ylabel('Fidelity / 100');panel.grid(alpha=.15);panel.margins(x=.18,y=.22)
        handles=[Line2D([],[],marker='o',color=c,markerfacecolor='white',linestyle='none',label=label) for c,label in [('#0072b2','ChatGPT · S = Sol, T = Terra'),('#76b900','NVIDIA · N = Nemotron'),('#7851a9','DeepSeek · D = Flash'),('#d97757','Claude · C = Opus or Sonnet')]]
        handles += [Line2D([],[],marker='o',color='#64748b',linestyle='none',label='Outer ring: custom agent')]
        if axis == 'cost_usd_per_asset': fig.suptitle('Inference cost of final retained outputs · retry spend reported separately',fontsize=12)
        fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
        for ext in ['svg','png']:fig.savefig(output/f'pareto-{suffix}.{ext}',dpi=160)
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('leaderboard',type=Path);p.add_argument('output',type=Path);a=p.parse_args();plot(json.loads(a.leaderboard.read_text()),a.output)
