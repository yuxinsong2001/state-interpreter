"""Archived descriptive alignment; no raw reads or causal diagnosis."""
import sys,json,hashlib
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root,out=map(Path,sys.argv[1:3]);assert not out.exists()
a=root/'results/2026-10-10_existing_ae_multiseed_audit/heldout_latents.csv'
b=root/'results/2026-10-10_relative_change_evaluation/state_traces.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(p.relative_to(root)):sha(p) for p in [a,b]}
features=pd.read_csv(a);states=pd.read_csv(b)
f=features[(features.test_bearing=='Bearing3_3')&(features.base_seed==20260918)]
s=states[(states.bearing=='Bearing3_3')&(states.seed==20260918)]
m=f.merge(s,left_on='step_id',right_on='step_id',how='inner',validate='one_to_one').sort_values('step_id')
assert len(m)==346 and m.step_id.min()==25
fig,axes=plt.subplots(5,1,figsize=(10,10),sharex=True)
cols=['rms_pair_mean','excess_kurtosis_pair_mean','level','trend','movement']
names=['Channel-mean RMS (source CSV units)','Channel-mean excess kurtosis (dimensionless)','Level (latent units)','Trend (latent units / step)','Movement (latent units / step)']
for ax,col,name in zip(axes,cols,names):
    ax.plot(m.step_id,m[col],color='#163b60',linewidth=1.2,label=name)
    ax.set_ylabel(col.replace('_pair_mean','').replace('excess_kurtosis','Kurtosis'),fontsize=9);ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=8)
axes[3].axhline(0,color='gray',linewidth=.7)
axes[-1].set_xlabel('Measurement step (zero-based; nominal 1-minute interval)')
fig.suptitle('Bearing3_3 | frozen AE base seed 20260918\nArchived signal features vs. causal relative states — descriptive, not health labels',fontsize=12)
fig.tight_layout(rect=[0,.03,1,.94])
fig.text(.02,.01,'Sources: existing_ae_multiseed_audit/heldout_latents.csv; relative_change_evaluation/state_traces.csv',fontsize=7)
assert all(sha(root/p)==v for p,v in hashes.items())
out.mkdir(parents=True);m.to_csv(out/'aligned_features_states.csv',index=False)
fig.savefig(out/'signal_state_alignment.png',dpi=180)
(out/'run_record.json').write_text(json.dumps(dict(status='completed',rows=len(m),bearing='Bearing3_3',base_seed=20260918,source_hashes=hashes,inputs_unchanged=True,raw_reads=0,training=False,interpretation='descriptive alignment only; no local waveform or spectral diagnosis'),indent=2),encoding='utf-8')
print('Aligned 346 archived observations; generated five-panel plot.')
