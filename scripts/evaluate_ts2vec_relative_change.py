"""Frozen compact TS2Vec pipeline comparison, no training or protected reads."""
import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
root,out=map(Path,sys.argv[1:3]);sys.path.insert(0,str(root/'src'))
from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler
from state_interpreter.encoders.ts2vec import TS2VecEncoder
from state_interpreter.ts2vec_experiment import causal_context_windows
from state_interpreter.relative_temporal import RelativeTemporalStateInterpreter
cfgpath=root/'configs/xjtu_condition3_ts2vec_development_v1.json'
cfg=json.loads(cfgpath.read_text(encoding='utf-8'));assert cfg['development_bearings']==['Bearing3_1','Bearing3_2','Bearing3_3']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(cfgpath):sha(cfgpath)}
for a in cfg['frozen_artifacts']:
    p=root/a['path'];assert sha(p)==a['sha256'];hashes[str(p)]=sha(p)
assert not out.exists();out.mkdir(parents=True);torch.set_num_threads(2)
sequences={}
for entry in cfg['input']['cache_entries']:
    b=entry['bearing_id'];assert b in cfg['development_bearings']
    p=root/cfg['input']['cache_directory']/f'{b}.npz';assert sha(p)==entry['sha256'];hashes[str(p)]=sha(p)
    with np.load(p,allow_pickle=False) as a:features=torch.tensor(a['features'],dtype=torch.float32)
    scaler=StableCausalFeatureScaler.fit(features,calibration_steps=15,mode='signed_log1p');sequences[b]=scaler.transform(features).numpy()
def encode(model,x):
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(x),64):
            windows=causal_context_windows(x,np.arange(start,min(start+64,len(x))),context_length=128)
            chunks.append(model(windows)[:,-1].numpy())
    return np.concatenate(chunks)
def interpret(z):
    m=RelativeTemporalStateInterpreter(embedding_dim=64,calibration_steps=15,temporal_window=10)
    return np.array([[np.nan]*3 if (o:=m.update(torch.tensor(v,dtype=torch.float64))) is None else o.state.numpy() for v in z])
states={};traces=[];checks=[]
original=root/cfg['output_directory']
arch=pd.read_csv(original/'per_seed_health_metrics.csv')
for seed in cfg['training']['seeds']:
    for fold in cfg['outer_lobo_folds']:
        b=fold['test'];p=original/f'seed_{seed}_{b}_final.pt';h=sha(p);hashes[str(p)]=h
        expected=arch[(arch.seed==seed)&(arch.test_bearing==b)].iloc[0].checkpoint_sha256;assert h==expected
        cp=torch.load(p,map_location='cpu',weights_only=True)
        assert cp['test_bearing']==b and cp['train_bearings']==fold['train'] and cp['seed']==seed and cp['iteration']==200
        enc_cfg=cfg['encoder'];encoder=TS2VecEncoder(**{k:enc_cfg[k] for k in ['input_dims','output_dims','hidden_dims','depth','mask_probability']})
        model=torch.optim.swa_utils.AveragedModel(encoder);model.load_state_dict(cp['state_dict']);model.eval()
        x=sequences[b];z=encode(model,x);assert np.isfinite(z).all()
        prefix=encode(model,x[:60]);err=float(np.max(np.abs(prefix-z[:60])));assert err<1e-5
        s=interpret(z);assert np.allclose(s[:60],interpret(z[:60]),equal_nan=True)
        states[(b,seed)]=s[25:]
        checks.append(dict(bearing=b,seed=seed,max_prefix_embedding_difference=err))
        for i in range(25,len(s)):traces.append(dict(bearing=b,seed=seed,step_id=i,level=s[i,0],trend=s[i,1],movement=s[i,2]))
        print(f'Frozen inference completed: {b}, seed {seed}',flush=True)
rows=[]
for b in cfg['development_bearings']:
    for a,c in itertools.combinations(cfg['training']['seeds'],2):
        x,y=states[(b,a)],states[(b,c)];sign=lambda v:np.where(np.abs(v)<=1e-9,0,np.sign(v))
        rows.append(dict(bearing=b,seed_a=a,seed_b=c,trend_sign_agreement=float(np.mean(sign(x[:,1])==sign(y[:,1]))),movement_rho=float(spearmanr(x[:,2],y[:,2]).statistic),level_rho=float(spearmanr(x[:,0],y[:,0]).statistic)))
ag=pd.DataFrame(rows);ag.to_csv(out/'seed_agreement.csv',index=False);pd.DataFrame(traces).to_csv(out/'state_traces.csv',index=False)
ae=pd.read_csv(root/'results/2026-10-10_relative_change_evaluation/seed_agreement.csv')
comparison=ag.merge(ae,on=['bearing','seed_a','seed_b'],suffixes=('_ts2vec','_ae'))
comparison.to_csv(out/'pipeline_comparison.csv',index=False)
assert hashes=={p:sha(Path(p)) for p in hashes}
passed=bool((ag.trend_sign_agreement>=.9).all() and (ag.movement_rho>=.8).all())
report=dict(status='completed',verification_status='ANALYZED',input_sha256=hashes,inputs_unchanged=True,prefix_checks=checks,stability_gate_passed=passed,decision='retain_pipeline_as_candidate' if passed else 'do_not_claim_stable_short_term_state',no_training=True,protected_bearings_not_read=['Bearing3_4','Bearing3_5'],limitations=['compact TS2Vec adaptation, not official full benchmark','feature65/history128/dim64 vs STFT/single measurement/dim8: pipeline comparison, not isolated objective ablation','engineering gates reused unchanged','same devices across seeds, no independent device inference','first15 feature scaling is allowed calibration; no emitted state before reference completion','no physical health labels'])
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(comparison.to_string(index=False));print('Gate passed:',passed)
