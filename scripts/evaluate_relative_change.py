"""Reuse existing radial interpreter; fixed synthetic and development checks."""
import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
root,out=map(Path,sys.argv[1:3]);sys.path.insert(0,str(root/'src'))
from state_interpreter.relative_temporal import RelativeTemporalStateInterpreter
source=root/'results/2026-10-10_existing_ae_multiseed_audit/heldout_latents.csv'
before=hashlib.sha256(source.read_bytes()).hexdigest()
assert not out.exists();out.mkdir(parents=True)
def run(z):
    model=RelativeTemporalStateInterpreter(embedding_dim=z.shape[1],calibration_steps=15,temporal_window=10)
    return np.array([[np.nan]*3 if (o:=model.update(torch.tensor(v,dtype=torch.float64))) is None else o.state.numpy() for v in z])
cal=np.zeros((15,2));t=np.linspace(0,2*np.pi,60)
cases={'stable':np.vstack([cal,np.zeros((60,2))]),'outward':np.vstack([cal,np.column_stack([np.arange(60),np.zeros(60)])]),'return':np.vstack([cal,np.column_stack([np.r_[np.arange(30),np.arange(29,-1,-1)],np.zeros(60)])]),'jump':np.vstack([cal,np.zeros((30,2)),np.tile([10.,0.],(30,1))]),'circle':np.vstack([cal,np.column_stack([np.cos(t),np.sin(t)])])}
checks={}
for name,z in cases.items():
    states=run(z)
    checks[name+'_causal_prefix']=bool(np.allclose(states[:50],run(z[:50]),equal_nan=True))
    checks[name+'_reset_repeat']=bool(np.allclose(states,run(z),equal_nan=True))
    if name=='stable':checks[name]=bool(np.allclose(states[15:],0))
    if name=='outward':checks[name]=bool(np.all(states[35:,1]>0))
    if name=='return':checks[name]=bool(np.all(states[60:,1]<0))
    if name=='jump':checks[name]=bool(states[45,2]==10 and states[44,2]==0 and states[-1,0]==10)
    if name=='circle':checks[name]=bool(np.max(np.abs(states[35:,1]))<1e-12 and np.all(states[35:,2]>0))
assert all(checks.values()),checks
df=pd.read_csv(source);assert set(df.test_bearing)=={'Bearing3_1','Bearing3_2','Bearing3_3'}
traces=[];correlations=[];groups={}
for (seed,b),g in df.groupby(['base_seed','test_bearing']):
    g=g.sort_values('step_id');z=g[[f'z_{d}' for d in range(8)]].to_numpy();state=run(z);state=state[25:];groups[(b,seed)]=state
    rms=g.rms_pair_mean.rolling(10,min_periods=1).mean().to_numpy()[25:]
    for j,name in enumerate(['level','trend','movement']):correlations.append(dict(bearing=b,seed=int(seed),component=name,rms_rho=float(spearmanr(state[:,j],rms).statistic),level_rho=float(spearmanr(state[:,j],state[:,0]).statistic)))
    for i,v in enumerate(state,25):traces.append(dict(bearing=b,seed=int(seed),step_id=i,level=v[0],trend=v[1],movement=v[2]))
agreement=[]
for b in sorted(df.test_bearing.unique()):
    seeds=sorted(s for bb,s in groups if bb==b)
    for a,c in itertools.combinations(seeds,2):
        x,y=groups[(b,a)],groups[(b,c)]
        sign=lambda v:np.where(np.abs(v)<=1e-9,0,np.sign(v))
        agreement.append(dict(bearing=b,seed_a=int(a),seed_b=int(c),trend_sign_agreement=float(np.mean(sign(x[:,1])==sign(y[:,1]))),movement_rho=float(spearmanr(x[:,2],y[:,2]).statistic)))
ag=pd.DataFrame(agreement)
passed=bool((ag.trend_sign_agreement>=.9).all() and (ag.movement_rho>=.8).all())
for name,data in [('state_traces.csv',traces),('component_correlations.csv',correlations),('seed_agreement.csv',agreement)]:pd.DataFrame(data).to_csv(out/name,index=False)
assert before==hashlib.sha256(source.read_bytes()).hexdigest()
report=dict(status='completed',synthetic_checks=checks,source_sha256=before,inputs_unchanged=True,decision='retain_as_stability_candidate' if passed else 'stop_stable_cross_seed_claim',stability_gate_passed=passed,no_training=True,no_protected_read=True,limitations=['engineering thresholds, not statistical significance','no physical labels','synthetic geometry does not prove useful machine semantics','different seeds share bearings; no independent-device inference','no new scaling or calibration quality gate implemented'])
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(ag.to_string(index=False));print(json.dumps(report,indent=2))
