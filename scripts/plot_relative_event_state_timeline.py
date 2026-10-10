"""Plot archived relative-event state messages without replay or model access."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

root=Path(sys.argv[1]).resolve()
source=root/'results/2026-10-09_relative_event_state_interface'
out=root/'results/2026-10-10_relative_event_state_timelines_final'
if out.exists():raise FileExistsError(out)
paths=[source/'states.csv',source/'episode_terminations.csv']
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with paths[0].open(encoding='utf-8') as f:states=list(csv.DictReader(f))
with paths[1].open(encoding='utf-8') as f:terms=list(csv.DictReader(f))
assert len(states)==10810 and len(terms)==19
out.mkdir()
navy,green,orange='#17365d','#71a99f','#bc6123'
plots=[]

def plot(bearing,feature,low=None,high=None,suffix='full'):
    series=[r for r in states if r['bearing_id']==bearing and r['provenance']==feature+'_episodic_v2']
    events=[r for r in terms if r['bearing_id']==bearing and r['feature']==feature]
    low=int(series[0]['step_id']) if low is None else low
    high=int(series[-1]['step_id']) if high is None else high
    selected=[r for r in series if low<=int(r['step_id'])<=high]
    x=np.array([int(r['step_id']) for r in selected])
    def nums(field):return np.array([float(r[field]) if r[field] else np.nan for r in selected])
    deviation=nums('feature_deviation_z');trend=nums('feature_trend_z_per_measurement')
    active=np.array([1. if r['active_event']=='True' else 0. if r['active_event']=='False' else np.nan for r in selected])
    fig,axes=plt.subplots(3,1,figsize=(12,8.8),sharex=True,
                          gridspec_kw={'height_ratios':[2,1.6,1.5]},constrained_layout=True)
    axes[0].plot(x,deviation,color=navy,lw=1,label='Feature deviation')
    axes[0].axhline(2,color=orange,ls='--',lw=1,label='Fixed threshold: 2 SD')
    axes[0].set_ylabel('Deviation (reference SD)')
    axes[1].plot(x,trend,color=navy,lw=1)
    axes[1].axhline(0,color='#7e8b97',ls='--',lw=1)
    axes[1].set_ylabel('Trend (reference SD / measurement)')
    if suffix=='full':
        axes[0].set_yscale('symlog',linthresh=2)
        axes[0].set_ylabel('Deviation (reference SD; symlog)')
        axes[1].set_yscale('symlog',linthresh=.1)
        axes[1].set_ylabel('Trend (SD / measurement; symlog)')
    axes[2].step(x,active,where='post',color=navy,lw=1.1)
    axes[2].set_yticks([0,1],['Monitoring','Active'])
    axes[2].set_ylabel('Candidate-event state')
    axes[2].set_ylim(-.18,2.15)
    for ax in axes:
        ax.set_xlim(low,high)
        ax.grid(alpha=.2)
        if low<15:ax.axvspan(max(0,low),min(14,high),color='#dfe5ed',alpha=.8)
    for e in events:
        start=int(e['confirm_step']);last=int(e['last_observed_active_step'])
        closed=e['right_censored']=='False'
        ending=int(e['termination_observed_step'])
        if last<low or start>high:continue
        for ax in axes:ax.axvspan(max(start,low),min(last+1,high),color=green,alpha=.15)
        # On full plots, avoid text collisions from densely packed short events.
        if suffix!='full' or last-start>=(high-low)*.035:
            label=f'E{e["episode_id"]}\n{e["observed_active_measurements"]} steps'+('+' if not closed else '')
            anchor=max(start,low)
            right=anchor>low+.85*(high-low)
            anchor=min(anchor,high-.01*(high-low))
            level=1.1+(int(e['episode_id'])%3)*.24 if suffix=='repeated_events' else 1.25
            axes[2].text(anchor,level,label,fontsize=8,ha='right' if right else 'left',va='bottom')
        if low<=ending<=high:
            axes[2].scatter([ending],[0 if closed else 1],s=35,marker='s' if closed else '^',color=orange,zorder=4)
    axes[0].legend(loc='upper left',fontsize=8)
    axes[2].legend(handles=[Patch(facecolor=green,alpha=.25,label='Active episode'),
                          Line2D([],[],marker='s',ls='',color=orange,label='Rule exit: five non-exceedances'),
                          Line2D([],[],marker='^',ls='',color=orange,label='Record end: right-censored')],
                   loc='upper left',fontsize=8,ncol=3)
    axes[2].set_xlabel('Measurement step ID (0-based; duration in observed measurements)')
    fig.suptitle(f'{bearing} | {feature.upper()} | relative event state timeline ({suffix})',fontsize=15)
    fig.supxlabel('Source: archived relative_event_state_interface CSVs. Exit does not establish physical recovery.\n'
                  'Trend uses up to 10 causal observations; missing values remain blank. "+" means duration is right-censored.',fontsize=9)
    filename=f'{bearing.lower()}_{feature}_{suffix}.png'
    fig.savefig(out/filename,dpi=170);plt.close(fig)
    plots.append({'file':filename,'bearing_id':bearing,'feature':feature,'first_step':low,'last_step':high,
                  'selected_rows':len(selected),'view':suffix})

for b in ('Bearing3_1','Bearing3_2','Bearing3_3'):
    for feat in ('rms','kurtosis'):plot(b,feat)
plot('Bearing3_1','rms',0,70,'early_event')
plot('Bearing3_2','rms',1200,1750,'repeated_events')
plot('Bearing3_2','rms',2300,2370,'late_reentry')
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert before==after
report={'status':'completed','input_sha256':before,'input_unchanged':True,'state_rows':len(states),
        'episode_records':len(terms),'plots':plots,'protected_not_read':['Bearing3_4','Bearing3_5'],
        'method':'archived CSV visualization; no state recomputation, training or parameter changes',
        'interpretation':'relative feature change and candidate events; no physical health ground truth'}
with (out/'plot_manifest.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
