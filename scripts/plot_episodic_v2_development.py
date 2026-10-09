"""Plot frozen replay CSVs only; never access bearing caches."""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
out = root / 'results/2026-10-09_condition3_episodic_v2_development'
with (out / 'trajectories.csv').open(encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
with (out / 'summary.csv').open(encoding='utf-8') as f:
    summary = list(csv.DictReader(f))
for strength in (False, True):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for i, bearing in enumerate(('Bearing3_1', 'Bearing3_2', 'Bearing3_3')):
        for j, feature in enumerate(('rms', 'kurtosis')):
            ax = axes[i,j]
            selected = [r for r in rows if r['bearing_id'] == bearing and r['provenance'] == feature+'_episodic_v2']
            info = next(s for s in summary if s['bearing_id']==bearing and s['feature']==feature)
            valid = [r for r in selected if r['active_event']!='']
            x = [int(r['step_id']) for r in valid]
            if strength:
                ax.plot(x, [float(r['deviation_z']) for r in valid], color='#17365d', lw=1, label='Standardized feature deviation')
                ax.axhline(2, color='#bd6424', ls='--', label='Threshold: 2 reference SD')
                ax.set_yscale('symlog', linthresh=2)
                ax.set_ylabel('Deviation (reference SD; symlog)')
            else:
                first = info['first_confirm_step']
                ax.step(x, [int(bool(first) and t>=int(first)) for t in x], where='post', color='#a9b2be', lw=3, label='v1: latched alarm')
                ax.step(x, [int(r['active_event']=='True') for r in valid], where='post', color='#17365d', lw=1.2, label='v2: active episode')
                ax.set_ylim(-0.1, 1.18)
                ax.set_yticks([0,1], ['Monitoring', 'Active'])
                ax.set_ylabel('Algorithmic event state')
            ax.axvspan(0,14,color='#e3e8ef',alpha=.8,label='Initial reference (15 steps)')
            ax.set_title(f'{bearing} | {feature.upper()} | {info["episode_count"]} v2 episodes',fontsize=11)
            ax.set_xlabel('Measurement step ID (0-based)')
            ax.grid(alpha=.2)
            if i==0 and j==0:
                ax.legend(fontsize=8, loc='upper left' if strength else 'lower right')
    fig.suptitle('Causal vibration-change episodes: '+('feature deviation' if strength else 'v1 versus v2'),fontsize=16)
    name = 'feature_deviation.png' if strength else 'v1_v2_event_comparison.png'
    path = out / name
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path,dpi=180)
    plt.close(fig)
print('Two figures created from frozen development results; no cache reads.')
