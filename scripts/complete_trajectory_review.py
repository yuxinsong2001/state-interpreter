"""Close archived AE review with fixed-window 8-D direction diagnostics."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
root,out=map(Path,sys.argv[1:3])
source=root/'results/2026-10-10_existing_ae_multiseed_audit/heldout_latents.csv'
before=hashlib.sha256(source.read_bytes()).hexdigest()
df=pd.read_csv(source);assert set(df.test_bearing)=={'Bearing3_1','Bearing3_2','Bearing3_3'}
assert not out.exists();out.mkdir(parents=True)
rows=[]
for (seed,b),g in df.groupby(['base_seed','test_bearing']):
    g=g.sort_values('step_id');z=g[[f'z_{i}' for i in range(8)]].to_numpy();n=len(z)
    # Fixed descriptive regions, not estimated change points or health stages.
    centers=[z[:15].mean(0),z[int(.4*n):int(.6*n)].mean(0),z[int(.8*n):int(.9*n)].mean(0),z[-15:].mean(0)]
    a=centers[1]-centers[0];bb=centers[2]-centers[1];c=centers[3]-centers[2]
    def cosine(x,y):return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
    rms=g.rms_pair_mean.to_numpy();dc=np.diff(z,axis=0);dr=np.diff(rms)
    # Within-trajectory affine association diagnostic only; not a deployable fit.
    X=np.column_stack([np.ones(n),np.log1p(rms)])
    fitted=X@np.linalg.lstsq(X,z,rcond=None)[0]
    r2=1-float(np.sum((z-fitted)**2)/np.sum((z-z.mean(0))**2))
    rows.append(dict(base_seed=int(seed),bearing=b,n=n,early_mid_vs_late_cosine=cosine(a,c),mid_prelate_vs_late_cosine=cosine(bb,c),early_mid_length=float(np.linalg.norm(a)),late_length=float(np.linalg.norm(c)),log_rms_affine_variance_fraction=r2,radial_vs_rms_rho=float(spearmanr(g.own_distance,rms).statistic)))
pd.DataFrame(rows).to_csv(out/'direction_metrics.csv',index=False)
assert before==hashlib.sha256(source.read_bytes()).hexdigest()
report=dict(status='completed',verification_status='ANALYZED',source_sha256=before,inputs_unchanged=True,windows={'initial':'first15','middle':'40–60%','prelate':'80–90%','end':'last15'},no_training=True,no_protected_data_read=True,limitations=['fixed retrospective windows use total length, not online phase detection','affine RMS association fitted on same trajectory, descriptive in-sample only','cosines within each model only; latent coordinates not comparable across separately trained models','no physical health labels; no causal claims'])
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
