"""Fixed-checkpoint readout bias and train-gradient diagnosis; no updates."""
import sys,csv,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
root,out=map(Path,sys.argv[1:3]);assert not out.exists()
sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'scripts'))
from train_gru_state_v1 import prepare,loss
from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
folder=root/'results/2026-10-10_gru_state_v1_formal'
paths=[folder/'state_outputs.csv',folder/'training_history.csv']+[folder/f'checkpoint_seed_{s}.pt' for s in [20260916,20260917,20260918]]
source=root/'runs/latent_analysis_z8_20260801/latent_trajectories.csv';paths.append(source)
hashes={str(p.relative_to(root)):sha(p) for p in paths}
df=pd.read_csv(paths[0]);history=pd.read_csv(paths[1]);segments=[];gradients=[];convergence=[]
for (seed,bearing),g in df.groupby(['seed','bearing']):
    g=g.sort_values('step')
    for idx,part in enumerate(np.array_split(np.arange(len(g)),3)):
        h=g.iloc[part];error=h.level.to_numpy()-h.target_level.to_numpy()
        segments.append(dict(seed=int(seed),bearing=bearing,segment=['early','middle','late'][idx],n=len(h),start_step=int(h.step.min()),end_step=int(h.step.max()),mean_signed_error=float(error.mean()),mae=float(abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),target_mean=float(h.target_level.mean())))
names=['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4'];rows={n:[] for n in names}
with source.open(newline='',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['bearing_id'] in rows:rows[r['bearing_id']].append(r)
raw={n:torch.tensor([[float(r[f'z_{i}']) for i in range(8)] for r in sorted(rows[n],key=lambda r:int(r['step_id']))]) for n in names}
torch.set_num_threads(1)
for seed in [20260916,20260917,20260918]:
    cp=torch.load(folder/f'checkpoint_seed_{seed}.pt',weights_only=True)
    seq,target,mean,std,scale=prepare(raw,cp['config'])
    assert torch.equal(mean,cp['train_mean']) and torch.equal(std,cp['train_std']) and torch.equal(scale,cp['level_target_scale'])
    m=GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8);m.load_state_dict(cp['model_state_dict']);m.eval()
    params=list(m.core.gru.parameters())
    for label,subset in [(n,[n]) for n in names[:3]]+[('train_episode_equal_mean',names[:3])]:
        joint,pred,level=loss(m,subset,seq,target,scale,25)
        a=torch.cat([g.flatten() for g in torch.autograd.grad(pred,params,retain_graph=True)])
        b=torch.cat([g.flatten() for g in torch.autograd.grad(level,params)])
        an=float(a.norm());bn=float(b.norm());den=an*bn
        gradients.append(dict(seed=seed,scope=label,prediction_loss=float(pred.detach()),level_loss=float(level.detach()),prediction_gradient_norm=an,level_gradient_norm=bn,level_over_prediction_norm=bn/an if an>0 else None,gradient_cosine=float(torch.dot(a,b)/den) if den>0 else None))
    h=history[history.seed==seed].sort_values('epoch');first=h[h.epoch>280];last=h[h.epoch>290]
    convergence.append(dict(seed=seed,last20_validation_joint_slope=float(np.polyfit(first.epoch,first.validation_joint,1)[0]),last10_validation_prediction_slope=float(np.polyfit(last.epoch,last.validation_prediction,1)[0]),last10_validation_level_slope=float(np.polyfit(last.epoch,last.validation_level,1)[0])))
assert all(sha(root/p)==h for p,h in hashes.items())
out.mkdir(parents=True)
for filename,data in [('segment_bias.csv',segments),('train_gradient_metrics.csv',gradients),('late_training_slopes.csv',convergence)]:pd.DataFrame(data).to_csv(out/filename,index=False)
(out/'report.json').write_text(json.dumps(dict(status='completed',input_hashes=hashes,inputs_unchanged=True,optimizer_steps=0,gradient_data='training episodes only',segmentation='equal-count thirds of step>=25; retrospective, not physical stages',excluded_embeddings_loaded=0,limitations=['gradient cosine at one frozen checkpoint is local, not causal proof','validation selected checkpoint','no health labels','late loss slope is not a convergence guarantee']),indent=2),encoding='utf-8')
print(pd.DataFrame(segments).query("bearing=='Bearing1_4'").to_string(index=False))
print(pd.DataFrame(gradients).query("scope=='train_episode_equal_mean'").to_string(index=False))
print(pd.DataFrame(convergence).to_string(index=False))
print('All per-episode gradient cosines:',[r['gradient_cosine'] for r in gradients])
