"""GRU-only dual-head development training; immutable configuration/output."""
import argparse,csv,json,hashlib,sys,time,copy
from pathlib import Path
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(raw,cfg):
    c,w,start=cfg['calibration_steps'],cfg['temporal_window'],cfg['score_start']
    centered={};targets={}
    for name,z in raw.items():
        if len(z)<=start or not torch.isfinite(z).all():raise ValueError('invalid episode')
        centered[name]=z-z[:c].mean(0)
        distances=centered[name].norm(dim=1)
        target=torch.full((len(z),),float('nan'))
        for t in range(c,len(z)):target[t]=distances[max(c,t-w+1):t+1].mean()
        targets[name]=target
    train=cfg['train_bearings']
    pooled=torch.cat([centered[n][c:] for n in train])
    mean=pooled.mean(0);std=pooled.std(0,unbiased=False).clamp_min(1e-6)
    scale=torch.stack([targets[n][start:].square().mean() for n in train]).mean().sqrt()
    if not torch.isfinite(scale) or scale<=1e-8:raise ValueError('invalid training target scale')
    return {n:(z-mean)/std for n,z in centered.items()},targets,mean,std,scale

def loss(model,names,sequences,targets,scale,start):
    predictions=[];levels=[]
    for n in names:
        z=sequences[n];o=model(z[None])
        predictions.append((o.next_embedding_prediction[0,start-1:-1]-z[start:]).square().mean())
        levels.append((o.scaled_relative_level[0,start:,0]-targets[n][start:]/scale).square().mean())
    pred=torch.stack(predictions).mean();level=torch.stack(levels).mean()
    return pred+level,pred,level

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('No overwrite permitted')
    cfg=json.loads(a.config.read_text());config_hash=sha(a.config)
    assert set(cfg['train_bearings'])=={'Bearing1_1','Bearing1_2','Bearing1_3'} and cfg['validation_bearing']=='Bearing1_4'
    sys.path.insert(0,str(a.repo/'src'))
    from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
    source=a.repo/cfg['input_latents'];source_hash=sha(source)
    allowed=set(cfg['train_bearings']+[cfg['validation_bearing']]);rows={n:[] for n in allowed}
    with source.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['bearing_id'] in allowed:rows[r['bearing_id']].append(r)
    raw={}
    for n,r in rows.items():
        r.sort(key=lambda x:int(x['step_id']));assert [int(x['step_id']) for x in r]==list(range(len(r)))
        raw[n]=torch.tensor([[float(x[f'z_{i}']) for i in range(8)] for x in r])
    seq,target,mean,std,scale=prepare(raw,cfg)
    # Validation cannot change training statistics, even if its values change.
    changed=dict(raw);changed[cfg['validation_bearing']]=raw[cfg['validation_bearing']]*100
    check=prepare(changed,cfg)
    assert torch.equal(mean,check[2]) and torch.equal(std,check[3]) and torch.equal(scale,check[4])
    torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    model=GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=cfg['hidden_dim'])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history=[];best=float('inf');best_state=None;stale=0;started=time.monotonic()
    for epoch in range(1,cfg['max_epochs']+1):
        if time.monotonic()-started>cfg['hard_timeout_seconds']:raise TimeoutError('Declared timeout')
        model.train();optimizer.zero_grad();total,pr,lv=loss(model,cfg['train_bearings'],seq,target,scale,cfg['score_start']);total.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']);optimizer.step()
        model.eval()
        with torch.no_grad():valid,vp,vl=loss(model,[cfg['validation_bearing']],seq,target,scale,cfg['score_start'])
        values=[float(v) for v in [total,pr,lv,valid,vp,vl,norm]]
        if not all(torch.isfinite(torch.tensor(values))):raise ValueError('nonfinite training')
        history.append(dict(epoch=epoch,train_joint=values[0],train_prediction=values[1],train_level=values[2],validation_joint=values[3],validation_prediction=values[4],validation_level=values[5],gradient_norm=values[6]))
        if float(valid)<best-cfg['minimum_improvement']:best=float(valid);best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=cfg['patience']:break
    model.load_state_dict(best_state);model.eval();outputs=[]
    with torch.no_grad():
        for n,z in seq.items():
            o=model(z[None]);prefix=model(z[None,:35])
            torch.testing.assert_close(o.scaled_relative_level[:,:35],prefix.scaled_relative_level,atol=1e-6,rtol=1e-5)
            level=o.scaled_relative_level[0,:,0]*scale
            for t in range(cfg['score_start'],len(z)):
                segment=level[max(15,t-9):t+1];x=torch.arange(len(segment),dtype=level.dtype);x=x-x.mean()
                trend=float((x*(segment-segment.mean())).sum()/x.square().sum())
                movement=float((z[t]-o.next_embedding_prediction[0,t-1]).square().mean().sqrt())
                outputs.append(dict(bearing=n,step=t,level=float(level[t]),target_level=float(target[n][t]),trend=trend,movement=movement))
    assert sha(source)==source_hash and sha(a.config)==config_hash
    a.output.mkdir(parents=True)
    for name,data in [('training_history.csv',history),('state_outputs.csv',outputs)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    torch.save(dict(model_state_dict=best_state,config=cfg,train_mean=mean,train_std=std,level_target_scale=scale,best_epoch=best_epoch,input_sha256=source_hash),a.output/'checkpoint.pt')
    report=dict(status='completed',purpose=cfg['purpose'],best_epoch=best_epoch,best_validation_joint=best,target_scale=float(scale),sample_counts={n:len(z) for n,z in raw.items()},source_sha256=source_hash,config_sha256=config_hash,inputs_unchanged=True,train_statistics_validation_invariant=True,prefix_checks_passed=True,excluded_bearing_embeddings_loaded=0,real_raw_reads=0,limitations=['short integration training, not final model evidence','explicit relative-distance proxy, not health label','shared CSV scanned structurally, excluded embeddings never materialized'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));print(history)
if __name__=='__main__':main()
