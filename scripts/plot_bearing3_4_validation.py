"""Read completed validation outputs only; no raw-data or model access."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source=Path(sys.argv[1])
paths=[source/'states.csv',source/'episode_terminations.csv']
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with paths[0].open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
with paths[1].open(encoding='utf-8') as f:events=list(csv.DictReader(f))
out=source/'figures'
out.mkdir(exist_ok=False)
for feature in ('rms','kurtosis'):
    data=[r for r in rows if r['provenance']==feature+'_episodic_v2']
    x=np.array([int(r['step_id']) for r in data])
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(12,8),constrained_layout=True)
    for ax,key,label in zip(axes[:2],('feature_deviation_z','feature_trend_z_per_measurement'),('Deviation (reference SD)','Trend (SD / measurement)')):
        y=[float(r[key]) if r[key] else np.nan for r in data]
        ax.plot(x,y,color='#17365d',lw=1)
        ax.set_ylabel(label+'; symlog')
        ax.set_yscale('symlog',linthresh=2 if ax is axes[0] else .1)
    axes[0].axhline(2,color='#bc6123',ls='--',label='Frozen threshold: 2 SD')
    axes[0].legend()
    axes[1].axhline(0,color='gray',ls='--')
    active=[1 if r['active_event']=='True' else 0 if r['active_event']=='False' else np.nan for r in data]
    axes[2].step(x,active,where='post',color='#17365d')
    axes[2].set_yticks([0,1],['Monitoring','Active'])
    axes[2].set_ylabel('Candidate-event state')
    axes[2].set_ylim(-.15,1.45)
    for e in events:
        if e['feature']!=feature:continue
        start,last=int(e['confirm_step']),int(e['last_observed_active_step'])
        for ax in axes:ax.axvspan(start,last+1,color='#71a99f',alpha=.18)
        closed=e['right_censored']=='False'
        axes[2].scatter(int(e['termination_observed_step']),0 if closed else 1,marker='s' if closed else '^',color='#bc6123')
    for ax in axes:
        ax.grid(alpha=.2);ax.set_xlim(x[0],x[-1]);ax.axvspan(0,14,color='gray',alpha=.15)
    axes[2].set_xlabel('Measurement step ID (0-based)')
    fig.suptitle('Bearing3_4 | '+feature.upper()+' | frozen episodic v2 validation')
    fig.supxlabel('Source: completed validation CSVs. Shading: active event; square: rule exit; triangle: record-end censoring.\nRelative vibration change only; no physical fault/recovery ground truth.',fontsize=9)
    fig.savefig(out/(feature+'_timeline.png'),dpi=170);plt.close(fig)
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with (out/'manifest.json').open('x',encoding='utf-8') as f:json.dump({'input_sha256':hashes,'input_unchanged':True,'method':'completed CSV visualization only'},f,indent=2)
