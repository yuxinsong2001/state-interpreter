"""Matched post-calibration readout diagnostics, archived development latents only."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root,source,out=map(Path,sys.argv[1:4]);sys.path.insert(0,str(root/'src'))
from state_interpreter.signed_axis import signed_level,causal_mean
assert not out.exists()
inputs=[source/'heldout_latents.csv',source/'metrics.csv']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(p):sha(p) for p in inputs}
df=pd.read_csv(inputs[0]);assert len(df)==16215
assert set(df.test_bearing)=={'Bearing3_1','Bearing3_2','Bearing3_3'}
out.mkdir(parents=True)
rows=[];refs=[];traces=[];checkpoints=[]
for (seed,b),g in df.groupby(['base_seed','test_bearing'],sort=True):
    g=g.sort_values('step_id');steps=g.step_id.to_numpy();assert np.array_equal(steps,np.arange(len(g)))
    cp_path=root/f'results/2026-09-18_condition3_arm_b_encoder_lobo_seed_{seed}/{b}_checkpoint.pt'
    before=sha(cp_path);cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    assert cp['test']==b and set(cp['train'])==set(df.test_bearing)-{b}
    z=g[[f'z_{i}' for i in range(8)]].to_numpy();center=z[:15].mean(axis=0)
    radius=float(np.sqrt(np.mean(np.sum((z[:15]-center)**2,axis=1))))
    assert radius>0
    signed=signed_level(z,cp['axis'],calibration_steps=15,temporal_window=10)
    methods={'signed_axis':signed}
    for name,col in [('own_radial','own_distance'),('global_radial','global_distance'),('rms_baseline','rms_pair_mean')]:
        value=np.full(len(g),np.nan);value[15:]=causal_mean(g[col].to_numpy()[15:],10);methods[name]=value
    mask=steps>=25
    for name,values in methods.items():
        v=values[mask];assert np.isfinite(v).all()
        rows.append(dict(base_seed=int(seed),bearing=b,method=name,score_start_step=25,smoothing_window=10,n_scored=int(mask.sum()),time_rho=float(spearmanr(steps[mask],v).statistic),rms_rho=float(spearmanr(methods['rms_baseline'][mask],v).statistic),backward_fraction=float(np.mean(np.diff(v)<-1e-9))))
    axis=np.asarray(cp['axis']);axis/=np.linalg.norm(axis)
    for start in [15,30,45]:
        delta=z[start:start+15].mean(axis=0)-center
        shift=float(np.linalg.norm(delta));along=float(delta@axis)
        refs.append(dict(base_seed=int(seed),bearing=b,window_start_step=start,window_end_step=start+14,calibration_radius=radius,center_shift_in_reference_radii=shift/radius,signed_center_shift=along,axis_cosine=along/shift if shift>0 else 0,rms_ratio_to_first15=float(g.rms_pair_mean.iloc[start:start+15].mean()/g.rms_pair_mean.iloc[:15].mean())))
    for i in np.flatnonzero(mask):traces.append(dict(base_seed=int(seed),bearing=b,step_id=int(i),**{name:float(vals[i]) for name,vals in methods.items()}))
    assert before==sha(cp_path);checkpoints.append(dict(path=str(cp_path),sha256=before))
assert hashes=={str(p):sha(p) for p in inputs}
for filename,data in [('matched_metrics.csv',rows),('reference_windows.csv',refs),('matched_traces.csv',traces)]:pd.DataFrame(data).to_csv(out/filename,index=False)
stats=pd.DataFrame(rows).groupby(['bearing','method']).time_rho.agg(['min','max','mean'])
stats.to_csv(out/'rho_ranges.csv')
fig,axes=plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
for ax,(seed,g) in zip(axes,pd.DataFrame(traces).query("bearing=='Bearing3_3'").groupby('base_seed')):
    for name in ['signed_axis','own_radial','global_radial']:
        vals=g[name].to_numpy();scale=np.std(vals)
        ax.plot(g.step_id,(vals-vals.mean())/scale,label=name)
    ax.set_ylabel('Standardized readout (unitless)');ax.set_title(f'Bearing3_3 | base seed {seed}');ax.grid(alpha=.2);ax.legend()
axes[-1].set_xlabel('Measurement step (1-minute intervals)')
fig.suptitle('Matched readouts: same 15-step reference, 10-step smoothing, score from step 25\nSource: matched_traces.csv; display standardization only')
fig.savefig(out/'bearing3_3_matched_readouts.png',dpi=160);plt.close(fig)
report=dict(status='completed',verification_status='ANALYZED',input_sha256=hashes,checkpoints=checkpoints,inputs_unchanged=True,no_training=True,no_raw_data_read=True,excluded=['Bearing3_4','Bearing3_5'],reference_windows='fixed 15-step windows [15,30), [30,45), [45,60), descriptive only',limitations=['post-hoc on three previously used development bearings','seed repetitions are not independent devices','radial change need not represent degradation','initial window has no certified health labels','reference shift normalized by within-window RMS latent radius, not a significance test','no model or parameter selected from these results'])
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(stats.to_string());print(pd.DataFrame(refs).groupby('bearing').center_shift_in_reference_radii.agg(['min','max']).to_string())
