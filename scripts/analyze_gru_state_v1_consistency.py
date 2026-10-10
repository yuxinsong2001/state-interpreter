"""Frozen GRU v1 trajectory analysis, not health accuracy evaluation."""
import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root,out=map(Path,sys.argv[1:3]);assert not out.exists()
src=root/'results/2026-10-10_gru_state_v1_formal/state_outputs.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(src);df=pd.read_csv(src)
assert set(df.bearing)=={'Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4'}
assert set(df.seed)=={20260916,20260917,20260918}
metrics=[];top=[]
out.mkdir(parents=True)
for bearing,group in df.groupby('bearing'):
    series={seed:g.sort_values('step') for seed,g in group.groupby('seed')}
    for a,b in itertools.combinations(sorted(series),2):
        x,y=series[a],series[b];assert np.array_equal(x.step,y.step)
        assert np.allclose(x.target_level,y.target_level,rtol=0,atol=0)
        near=(abs(x.trend.to_numpy())<=1e-9)|(abs(y.trend.to_numpy())<=1e-9)
        k=max(1,int(np.ceil(len(x)*.1)))
        ix=set(x.nlargest(k,'movement').step);iy=set(y.nlargest(k,'movement').step)
        metrics.append(dict(bearing=bearing,seed_a=int(a),seed_b=int(b),n=len(x),
            level_rho=float(spearmanr(x.level,y.level).statistic),
            level_pair_rmse=float(np.sqrt(np.mean((x.level.to_numpy()-y.level.to_numpy())**2))),
            trend_rho=float(spearmanr(x.trend,y.trend).statistic),
            trend_sign_agreement=float(np.mean(np.sign(x.trend.to_numpy())==np.sign(y.trend.to_numpy()))),
            near_zero_pair_fraction=float(near.mean()),
            movement_rho=float(spearmanr(x.movement,y.movement).statistic),
            movement_top_decile_overlap=float(len(ix&iy)/k),
            movement_peak_a=int(x.loc[x.movement.idxmax(),'step']),movement_peak_b=int(y.loc[y.movement.idxmax(),'step'])))
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for seed,g in series.items():
        for ax,col in zip(axes,['level','trend','movement']):
            ax.plot(g.step,g[col],linewidth=1,label=str(seed))
        row=g.loc[g.movement.idxmax()]
        top.append(dict(bearing=bearing,seed=int(seed),movement_peak_step=int(row.step),movement_peak=float(row.movement)))
    reference=next(iter(series.values()))
    axes[0].plot(reference.step,reference.target_level,color='black',linestyle='--',linewidth=1,label='Relative proxy target')
    for ax,label in zip(axes,['Level (latent distance units)','Trend (latent units / step)','Movement (standardized residual RMS)']):
        ax.set_ylabel(label,fontsize=8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1].axhline(0,color='gray',linewidth=.6)
    axes[-1].set_xlabel('Measurement step (zero-based; nominal 1-minute interval)')
    fig.suptitle(f'GRU v1 | {bearing} | frozen three-seed outputs\n'+('Training episode' if bearing!='Bearing1_4' else 'Validation episode used for checkpoint selection'))
    fig.tight_layout(rect=[0,.03,1,.94]);fig.text(.02,.01,'Source: results/2026-10-10_gru_state_v1_formal/state_outputs.csv; no health labels',fontsize=8)
    fig.savefig(out/f'{bearing}_gru_states.png',dpi=180);plt.close(fig)
pd.DataFrame(metrics).to_csv(out/'pair_metrics.csv',index=False);pd.DataFrame(top).to_csv(out/'movement_peaks.csv',index=False)
assert before==sha(src)
(out/'report.json').write_text(json.dumps(dict(status='completed',input_sha256=before,inputs_unchanged=True,training=False,new_bearing_reads=False,numerical_zero_tolerance=1e-9,top_decile='retrospective within-episode ranking; not event threshold',limitations=['three seeds share same devices','validation selected weights','rank correlation not health accuracy','no preset reliability pass/fail gate']),indent=2),encoding='utf-8')
print(pd.DataFrame(metrics).to_string(index=False))
