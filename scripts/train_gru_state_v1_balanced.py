"""Three-seed shared-gradient-balanced GRU development training."""
import argparse,csv,json,sys,time,copy,math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
from train_gru_state_v1 import prepare,loss,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),'Refuse overwrite'
    cfg=json.loads(a.config.read_text());ch=sha(a.config);source=a.repo/cfg['input_latents'];sh=sha(source)
    names=cfg['train_bearings']+[cfg['validation_bearing']]
    assert names==['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4']
    sys.path.insert(0,str(a.repo/'src'))
    from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
    from state_interpreter.gru_task_balance import balanced_shared_gradients
    rows={n:[] for n in names}
    with source.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['bearing_id'] in rows:rows[r['bearing_id']].append(r)
    raw={}
    for n in names:
        r=sorted(rows[n],key=lambda r:int(r['step_id']));assert [int(x['step_id']) for x in r]==list(range(len(r)))
        raw[n]=torch.tensor([[float(x[f'z_{i}']) for i in range(8)] for x in r])
    assert cfg['shared_gradient_balance'] is True
    seq,target,mean,std,scale=prepare(raw,cfg);torch.set_num_threads(1)
    changed=dict(raw);changed['Bearing1_4']=raw['Bearing1_4']*100
    check=prepare(changed,cfg);assert all(torch.equal(x,y) for x,y in zip([mean,std,scale],check[2:]))
    a.output.mkdir(parents=True);summary=[];histories=[];states=[];started=time.monotonic()
    for seed in cfg['seeds']:
        torch.manual_seed(seed);m=GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8)
        opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        best=math.inf;stale=0;best_state=None
        for epoch in range(1,cfg['max_epochs']+1):
            if time.monotonic()-started>cfg['hard_timeout_seconds']:raise TimeoutError('Declared timeout')
            m.train();opt.zero_grad();total,pr,lv=loss(m,names[:3],seq,target,scale,25);balance=balanced_shared_gradients(m,pr,lv)
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();m.eval()
            with torch.no_grad():v,vp,vl=loss(m,names[3:],seq,target,scale,25)
            values=[float(x.detach()) for x in [total,pr,lv,v,vp,vl]]
            assert all(math.isfinite(x) for x in values)
            histories.append(dict(seed=seed,epoch=epoch,train_joint=values[0],train_prediction=values[1],train_level=values[2],validation_joint=values[3],validation_prediction=values[4],validation_level=values[5],prediction_gradient_norm=balance.prediction_norm,level_gradient_norm=balance.level_norm,prediction_weight=balance.prediction_weight,level_weight=balance.level_weight,gradient_cosine=balance.cosine,fallback=balance.fallback))
            if values[3]<best-cfg['minimum_improvement']:best=values[3];best_epoch=epoch;best_state=copy.deepcopy(m.state_dict());stale=0
            else:stale+=1
            if stale>=cfg['patience']:break
        m.load_state_dict(best_state);m.eval();path=a.output/f'checkpoint_seed_{seed}.pt'
        torch.save(dict(model_state_dict=best_state,config=cfg,train_mean=mean,train_std=std,level_target_scale=scale,best_epoch=best_epoch,input_sha256=sh),path)
        loaded=torch.load(path,weights_only=True);fresh=GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8);fresh.load_state_dict(loaded['model_state_dict']);fresh.eval()
        with torch.no_grad():
            for n in names:
                z=seq[n];o=m(z[None]);r=fresh(z[None]);prefix=fresh(z[None,:35])
                for left,right in [(o.next_embedding_prediction,r.next_embedding_prediction),(o.scaled_relative_level,r.scaled_relative_level)]:
                    assert torch.equal(left,right)
                torch.testing.assert_close(r.next_embedding_prediction[:,:35],prefix.next_embedding_prediction,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(r.scaled_relative_level[:,:35],prefix.scaled_relative_level,atol=1e-6,rtol=1e-5)
                forecast=(r.next_embedding_prediction[0,24:-1]-z[25:]).square().mean()
                level=r.scaled_relative_level[0,:,0]*scale
                level_loss=((level[25:]-target[n][25:])/scale).square().mean()
                joint,lp,ll=loss(fresh,[n],seq,target,scale,25)
                torch.testing.assert_close(forecast,lp);torch.testing.assert_close(level_loss,ll)
                summary.append(dict(seed=seed,bearing=n,best_epoch=best_epoch,stop_epoch=epoch,forecast_mse=float(forecast),level_nrmse=float(level_loss.sqrt()),level_rmse=float(level_loss.sqrt()*scale),checkpoint_sha256=sha(path),reload_exact=True))
                for t in range(25,len(z)):
                    segment=level[max(15,t-9):t+1];x=torch.arange(len(segment),dtype=level.dtype);x-=x.mean()
                    trend=float((x*(segment-segment.mean())).sum()/x.square().sum())
                    residual=float((z[t]-r.next_embedding_prediction[0,t-1]).square().mean().sqrt())
                    states.append(dict(seed=seed,bearing=n,step=t,level=float(level[t]),target_level=float(target[n][t]),trend=trend,movement=residual))
        print(f'seed {seed}: best epoch {best_epoch}, stop {epoch}, validation joint {best:.6f}',flush=True)
    assert sha(source)==sh and sha(a.config)==ch
    for name,data in [('training_history.csv',histories),('metrics.csv',summary),('state_outputs.csv',states)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    report=dict(status='completed',source_sha256=sh,config_sha256=ch,inputs_unchanged=True,checkpoint_reload_exact=True,prefix_checks_passed=True,loss_alignment_verified=True,shared_gradient_balance=True,train_statistics_validation_invariant=True,embedding_bearings_loaded=names,raw_reads=0,protected_directory_reads=0,level_target_scale=float(scale),limitations=['development only; validation selected checkpoints','Level learns relative-distance proxy, not health','excluded rows scanned structurally but embeddings not materialized'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(summary)
if __name__=='__main__':main()
