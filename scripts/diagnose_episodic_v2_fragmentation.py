"""Descriptive diagnosis from archived CSVs; no model/cache access or rule changes."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]).resolve()
source = root / 'results/2026-10-09_condition3_episodic_v2_development'
output = root / 'results/2026-10-09_episodic_v2_fragmentation_diagnosis'
if output.exists():
    raise FileExistsError(output)
paths = [source / p for p in ('trajectories.csv', 'episodes.csv', 'summary.csv')]
inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with paths[0].open(encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
with paths[1].open(encoding='utf-8') as f:
    episodes = list(csv.DictReader(f))
with paths[2].open(encoding='utf-8') as f:
    summaries = list(csv.DictReader(f))
diagnostics, gaps, groups = [], [], []
for summary in summaries:
    bearing, feature = summary['bearing_id'], summary['feature']
    series = [r for r in rows if r['bearing_id']==bearing and r['provenance']==feature+'_episodic_v2']
    events = [e for e in episodes if e['bearing_id']==bearing and e['feature']==feature]
    for index, event in enumerate(events):
        start = int(event['confirm_step'])
        end = int(event['exit_confirm_step']) if event['exit_confirm_step'] else int(series[-1]['step_id'])+1
        active = [r for r in series if start<=int(r['step_id'])<end]
        z = np.array([float(r['deviation_z']) for r in active])
        trend = np.array([float(r['trend']) for r in active])
        assert len(active)==int(event['active_measurements'])
        assert all(r['active_event']=='True' for r in active)
        diag = {**event, 'median_deviation_z': float(np.median(z)),
                'q25_deviation_z': float(np.quantile(z,.25)), 'q75_deviation_z': float(np.quantile(z,.75)),
                'fraction_above_threshold': float(np.mean(z>2)),
                'median_trend_z_per_step': float(np.median(trend)),
                'raw_threshold_crossings_during_active': int(np.count_nonzero(np.diff(z>2)))}
        if event['exit_confirm_step']:
            confirm = int(event['exit_confirm_step'])
            window = [r for r in series if confirm-4<=int(r['step_id'])<=confirm]
            vals = [float(r['deviation_z']) for r in window]
            assert len(vals)==5 and max(vals)<=2+1e-12
            diag.update(exit_five_min_z=min(vals), exit_five_max_z=max(vals), exit_five_mean_z=float(np.mean(vals)))
        else:
            diag.update(exit_five_min_z=None, exit_five_max_z=None, exit_five_mean_z=None)
        diagnostics.append(diag)
        if index+1<len(events):
            nxt=events[index+1]
            candidate=int(nxt['candidate_start_step'])
            gaprows=[r for r in series if end<=int(r['step_id'])<candidate]
            gaps.append({'bearing_id':bearing,'feature':feature,'after_episode':int(event['episode_id']),
                         'exit_confirm_step':end,'next_candidate_step':candidate,
                         'monitoring_steps_before_next_exceedance_run':candidate-end,
                         'median_gap_deviation_z':float(np.median([float(r['deviation_z']) for r in gaprows])) if gaprows else None})
    groups.append({'bearing_id':bearing,'feature':feature,'episode_count':len(events),
                   'completed_durations':[int(e['active_measurements']) for e in events if e['exit_confirm_step']],
                   'right_censored_durations':[int(e['active_measurements']) for e in events if not e['exit_confirm_step']]})

output.mkdir()
for name, records in [('episode_diagnostics.csv',diagnostics),('inter_episode_gaps.csv',gaps)]:
    with (output/name).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
report={'status':'ANALYZED','input_sha256':inputs,'episode_count':len(diagnostics),'groups':groups,
        'protected_not_read':['Bearing3_4','Bearing3_5'],
        'interpretation':'post-hoc descriptive trajectory diagnosis; no physical event accuracy',
        'rules_unchanged':True,'retrained':False}
with (output/'report.json').open('x',encoding='utf-8') as f:
    json.dump(report,f,indent=2,allow_nan=False)

series=[r for r in rows if r['bearing_id']=='Bearing3_2' and r['provenance']=='rms_episodic_v2' and r['deviation_z']]
fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
for ax,(low,high,title) in zip(axes,[(15,2495,'Full monitored record'),(1200,1750,'Repeated entries and exits'),(2300,2370,'Exit from the long episode and re-entry')]):
    selected=[r for r in series if low<=int(r['step_id'])<=high]
    x=np.array([int(r['step_id']) for r in selected]);z=np.array([float(r['deviation_z']) for r in selected])
    ax.plot(x,z,color='#17365d',lw=1,label='RMS deviation')
    ax.axhline(2,color='#bc6225',ls='--',label='Fixed threshold: 2 SD')
    for e in episodes:
        if e['bearing_id']=='Bearing3_2' and e['feature']=='rms':
            a=int(e['confirm_step']);b=int(e['exit_confirm_step']) if e['exit_confirm_step'] else 2496
            if b>=low and a<=high:
                ax.axvspan(max(a,low),min(b,high),color='#73aaa2',alpha=.22)
                ax.text(max(a,low),.95,str(e['episode_id']),transform=ax.get_xaxis_transform(),fontsize=8)
    ax.set_title('Bearing3_2 RMS — '+title,fontsize=12)
    ax.set_xlabel('Measurement step ID (0-based)');ax.set_ylabel('Deviation (reference SD)');ax.grid(alpha=.2)
    if low==15:ax.set_yscale('symlog',linthresh=2);ax.set_ylabel('Deviation (reference SD; symlog)');ax.legend(fontsize=9)
fig.suptitle('Event fragmentation diagnosis | shaded regions: v2 active episodes',fontsize=15)
fig.savefig(output/'bearing3_2_fragmentation.png',dpi=180);plt.close(fig)
print(json.dumps({'groups':groups,'b3_2_rms_episodes':[d for d in diagnostics if d['bearing_id']=='Bearing3_2' and d['feature']=='rms'],
                  'b3_2_rms_gaps':[g for g in gaps if g['bearing_id']=='Bearing3_2' and g['feature']=='rms']},indent=2))
