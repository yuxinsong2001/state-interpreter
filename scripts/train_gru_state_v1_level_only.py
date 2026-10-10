"""Level-only objective ablation of the fixed dual-head GRU architecture."""
import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from train_gru_state_v1 import prepare, loss, sha


def main():
    parser = argparse.ArgumentParser()
    for name in ('repo', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Refuse overwrite'
    cfg = json.loads(args.config.read_text())
    assert cfg['level_only'] is True
    assert cfg['checkpoint_selection'] == 'validation_joint_matched_control'
    assert cfg['calibration_steps'] == 15 and cfg['temporal_window'] == 10 and cfg['score_start'] == 25
    assert cfg['seeds'] == [20260916, 20260917, 20260918]
    names = cfg['train_bearings'] + [cfg['validation_bearing']]
    assert names == ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4']
    sys.path.insert(0, str(args.repo/'src'))
    from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
    source = args.repo/cfg['input_latents']
    input_hash, config_hash = sha(source), sha(args.config)
    rows = {name: [] for name in names}
    with source.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['bearing_id'] in rows:
                rows[row['bearing_id']].append(row)
    raw = {}
    for name, data in rows.items():
        data.sort(key=lambda row: int(row['step_id']))
        assert [int(row['step_id']) for row in data] == list(range(len(data)))
        raw[name] = torch.tensor([[float(row[f'z_{i}']) for i in range(8)] for row in data])
    seq, target, mean, std, scale = prepare(raw, cfg)
    changed = dict(raw)
    changed['Bearing1_4'] = raw['Bearing1_4']*100
    check = prepare(changed,cfg)
    assert all(torch.equal(x,y) for x,y in zip((mean,std,scale),check[2:]))
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    history, metrics, states = [], [], []
    started = time.monotonic()
    for seed in cfg['seeds']:
        torch.manual_seed(seed)
        model = GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8)
        head_initial = copy.deepcopy(model.core.prediction_head.state_dict())
        for parameter in model.core.prediction_head.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        best, stale = math.inf, 0
        for epoch in range(1,cfg['max_epochs']+1):
            if time.monotonic()-started > cfg['hard_timeout_seconds']:
                raise TimeoutError('Declared experiment timeout')
            model.train()
            optimizer.zero_grad()
            joint,pred,level = loss(model,names[:3],seq,target,scale,25)
            level.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            model.eval()
            with torch.no_grad(): val,vp,vl = loss(model,names[3:],seq,target,scale,25)
            values = [float(x.detach()) for x in (joint,pred,level,val,vp,vl)]
            assert all(math.isfinite(x) for x in values)
            assert all(p.grad is None for p in model.core.prediction_head.parameters())
            assert all(torch.equal(head_initial[k],v) for k,v in model.core.prediction_head.state_dict().items())
            history.append(dict(seed=seed,epoch=epoch,train_joint=values[0],train_prediction=values[1],train_level=values[2],validation_joint=values[3],validation_prediction=values[4],validation_level=values[5]))
            if values[3] < best-cfg['minimum_improvement']:
                best,best_epoch,best_state,stale = values[3],epoch,copy.deepcopy(model.state_dict()),0
            else: stale += 1
            if stale >= cfg['patience']: break
        model.load_state_dict(best_state)
        checkpoint = args.output/f'checkpoint_seed_{seed}.pt'
        torch.save(dict(model_state_dict=best_state,config=cfg,train_mean=mean,train_std=std,level_target_scale=scale,best_epoch=best_epoch,input_sha256=input_hash),checkpoint)
        fresh = GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8)
        fresh.load_state_dict(torch.load(checkpoint,weights_only=True)['model_state_dict'])
        fresh.eval()
        with torch.no_grad():
            for name in names:
                z = seq[name]
                original,reloaded,prefix = model(z[None]),fresh(z[None]),fresh(z[None,:35])
                for field in ('next_embedding_prediction','scaled_relative_level'):
                    assert torch.equal(getattr(original,field),getattr(reloaded,field))
                    torch.testing.assert_close(getattr(reloaded,field)[:,:35],getattr(prefix,field),atol=1e-6,rtol=1e-5)
                pmse = (reloaded.next_embedding_prediction[0,24:-1]-z[25:]).square().mean()
                lmse = (reloaded.scaled_relative_level[0,25:,0]-target[name][25:]/scale).square().mean()
                _,lp,ll = loss(fresh,[name],seq,target,scale,25)
                torch.testing.assert_close(pmse,lp)
                torch.testing.assert_close(lmse,ll)
                metrics.append(dict(seed=seed,bearing=name,best_epoch=best_epoch,stop_epoch=epoch,forecast_mse=float(pmse),level_nrmse=float(lmse.sqrt()),level_rmse=float(lmse.sqrt()*scale),checkpoint_sha256=sha(checkpoint)))
                level_values = reloaded.scaled_relative_level[0,:,0]*scale
                for t in range(25,len(z)):
                    segment = level_values[max(15,t-9):t+1]
                    x = torch.arange(len(segment),dtype=segment.dtype)
                    x -= x.mean()
                    trend = float((x*(segment-segment.mean())).sum()/x.square().sum())
                    states.append(dict(seed=seed,bearing=name,step=t,level=float(level_values[t]),target_level=float(target[name][t]),trend=trend))
        print(f'seed {seed}: best={best_epoch}, stop={epoch}, validation_joint={best:.6f}',flush=True)
    assert sha(source)==input_hash and sha(args.config)==config_hash
    for name,data in [('metrics.csv',metrics),('training_history.csv',history),('state_outputs.csv',states)]:
        with (args.output/name).open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = dict(status='completed',experiment_kind=cfg['experiment_kind'],source_sha256=input_hash,config_sha256=config_hash,runner_sha256=sha(Path(__file__)),prediction_head_frozen_verified=True,inputs_unchanged=True,checkpoint_reload_exact=True,prefix_checks_passed=True,loss_alignment_verified=True,train_statistics_validation_invariant=True,embedding_bearings_loaded=names,raw_reads=0,protected_directory_reads=0,level_target_scale=float(scale),limitations=['development only; checkpoints selected on B1_4 with original joint criterion','relative Level proxy, not health truth','prediction is frozen zero-residual Persistence, not learned forecast; no Movement state emitted','excluded CSV rows scanned, embeddings not materialized'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__': main()
