"""Fixed archived-embedding reference sensitivity; no model or threshold search."""
import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
import torch
from state_interpreter.relative_temporal import RelativeTemporalStateInterpreter
root,out=map(Path,sys.argv[1:3]); assert not out.exists()
source=root/'results/2026-10-10_existing_ae_multiseed_audit/heldout_latents.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_hash=sha(source);df=pd.read_csv(source)
assert set(df.test_bearing)=={'Bearing3_1','Bearing3_2','Bearing3_3'}
assert set(df.base_seed)=={20260918,20260921,20260924}
curves=[];policies=[];pairs=[]
for (seed,bearing),g in df.groupby(['base_seed','test_bearing']):
    g=g.sort_values('step_id'); assert list(g.step_id)==list(range(len(g)))
    z=g[[f'z_{i}' for i in range(8)]].to_numpy(dtype=np.float64)
    states={};centers={};limits={}
    for n in [15,30,60]:
        model=RelativeTemporalStateInterpreter(embedding_dim=8,calibration_steps=n,temporal_window=10)
        records=[]
        for step,value in enumerate(z):
            state=model.update(torch.tensor(value,dtype=torch.float64))
            if state is not None:
                records.append(dict(step=step,level=float(state.level),trend=float(state.trend),movement=float(state.movement)))
        h=pd.DataFrame(records);centers[n]=model.baseline.numpy()
        fit=h[(h.step>=80)&(h.step<120)];assert len(fit)==40
        limit=float(np.quantile(abs(fit.trend),.8,method='linear'));limits[n]=limit
        mad=float(np.median(abs(fit.level-fit.level.median())))
        shift=float(fit.level.iloc[-10:].median()-fit.level.iloc[:10].median())
        rho=float(spearmanr(fit.step,fit.level).statistic)
        drift=abs(rho)>=.8 and abs(shift)>3*mad
        h['label']=np.where(abs(h.trend)<=limit,'within_resolution',np.where(h.trend>0,'moving_away','moving_toward'))
        eval=h[h.step>=140].copy();states[n]=eval
        policies.append(dict(seed=int(seed),bearing=bearing,reference_length=n,limit=limit,fit_rho=rho,drift_guard=bool(drift),hypothetical_coverage=float((eval.label!='within_resolution').mean()),evaluation_n=len(eval)))
        h['seed']=seed;h['bearing']=bearing;h['reference_length']=n;curves.append(h)
    radius=np.mean(np.linalg.norm(z[:15]-centers[15],axis=1))
    for a,b in itertools.combinations([15,30,60],2):
        x,y=states[a],states[b];assert np.array_equal(x.step,y.step)
        scale=float(np.median(abs(x.level)))
        pairs.append(dict(seed=int(seed),bearing=bearing,reference_a=a,reference_b=b,
            center_shift=float(np.linalg.norm(centers[a]-centers[b])),
            center_shift_over_first15_radius=float(np.linalg.norm(centers[a]-centers[b])/radius) if radius>0 else None,
            level_rho=float(spearmanr(x.level,y.level).statistic),
            level_median_abs_diff=float(np.median(abs(x.level.to_numpy()-y.level.to_numpy()))),
            level_relative_median_diff=float(np.median(abs(x.level.to_numpy()-y.level.to_numpy()))/scale) if scale>0 else None,
            trend_sign_agreement=float(np.mean(np.sign(x.trend.to_numpy())==np.sign(y.trend.to_numpy()))),
            hypothetical_label_agreement=float(np.mean(x.label.to_numpy()==y.label.to_numpy())),
            movement_max_abs_diff=float(np.max(abs(x.movement.to_numpy()-y.movement.to_numpy())))))
assert source_hash==sha(source)
out.mkdir(parents=True)
pd.concat(curves).to_csv(out/'state_traces.csv',index=False)
pd.DataFrame(policies).to_csv(out/'policy_metrics.csv',index=False)
pairdf=pd.DataFrame(pairs);pairdf.to_csv(out/'pair_metrics.csv',index=False)
(out/'run_record.json').write_text(json.dumps(dict(status='completed',source_sha256=source_hash,reference_lengths=[15,30,60],window=10,quantile=.8,common_fit=[80,120],common_check=[120,140],common_evaluation_start=140,raw_reads=0,protected_reads=0,training=False,inputs_unchanged=True,interpretation='hypothetical labels only; no accepted reference qualification'),indent=2),encoding='utf-8')
print(pairdf.groupby('bearing')[['center_shift_over_first15_radius','level_rho','level_relative_median_diff','trend_sign_agreement','hypothetical_label_agreement','movement_max_abs_diff']].agg(['min','max']).to_string())
print(pd.DataFrame(policies).groupby('bearing')[['hypothetical_coverage','drift_guard']].agg(['min','max']).to_string())
