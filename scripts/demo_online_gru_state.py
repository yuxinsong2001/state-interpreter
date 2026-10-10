"""Replay archived development embeddings through the online GRU wrapper."""
import argparse,csv,json,sys
from dataclasses import asdict
from pathlib import Path
import torch
from train_gru_state_v1 import sha

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'Refuse overwrite'
    sys.path.insert(0,str(args.repo/'src'))
    from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
    from state_interpreter.online_gru_state import OnlineGRUStateInterpreter
    run=args.repo/'results/2026-10-10_gru_state_v1_budget600'
    path=run/'checkpoint_seed_20260916.pt'
    loaded=torch.load(path,weights_only=True)
    model=GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8)
    model.load_state_dict(loaded['model_state_dict'])
    online=OnlineGRUStateInterpreter(model,loaded['train_mean'],loaded['train_std'],loaded['level_target_scale'])
    source=args.repo/'runs/latent_analysis_z8_20260801/latent_trajectories.csv'
    assert sha(source)==loaded['input_sha256']
    names=['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4']
    rows={n:[] for n in names}
    with source.open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['bearing_id'] in rows:rows[row['bearing_id']].append(row)
    with (run/'state_outputs.csv').open(newline='',encoding='utf-8') as stream:
        reference={(r['bearing'],int(r['step'])):r for r in csv.DictReader(stream) if int(r['seed'])==20260916}
    output=[];errors={k:0. for k in ('level','trend','movement')}
    for name in names:
        raw=torch.tensor([[float(r[f'z_{i}']) for i in range(8)] for r in sorted(rows[name],key=lambda r:int(r['step_id']))])
        online.reset()
        sequence=[online.update(z) for z in raw]
        online.reset()
        assert sequence==[online.update(z) for z in raw]
        for value in sequence:
            record=asdict(value);record['bearing']=name;output.append(record)
            if value.phase=='READY':
                baseline=reference[name,value.step]
                for field in errors:
                    actual=getattr(value,field);expected=float(baseline[field])
                    errors[field]=max(errors[field],abs(actual-expected))
                    assert abs(actual-expected)<=1e-5+1e-5*abs(expected),(name,value.step,field)
    args.output.mkdir(parents=True)
    with (args.output/'online_states.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    report=dict(status='completed',seed=20260916,checkpoint_sha256=sha(path),source_sha256=sha(source),bearings=names,total_rows=len(output),ready_rows=sum(r['phase']=='READY' for r in output),max_absolute_online_offline_error=errors,reset_replay_exact=True,limitations=['archived embedding replay, not live sensor deployment','B1_4 is development validation','known early overestimate and late underestimate retained','no health thresholds or maintenance decisions','excluded CSV rows scanned but embeddings not materialized'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    data=[r for r in output if r['bearing']=='Bearing1_4' and r['phase']=='READY']
    for axis,field,label in zip(axes,errors,['Level (relative proxy units)','Trend (proxy units / measurement)','Movement (standardized embedding RMSE)']):
        axis.plot([r['step'] for r in data],[r[field] for r in data],label='Online GRU')
        axis.set_ylabel(label);axis.grid(alpha=.25);axis.legend()
    axes[-1].set_xlabel('Measurement index (not seconds)')
    figure.suptitle('Bearing1_4: online relative GRU state — development replay')
    figure.text(.01,.01,'Source: ordinary GRU seed 20260916, archived XJTU-SY z8. Not physical health.',fontsize=9)
    figure.tight_layout(rect=(0,.03,1,.96));figure.savefig(args.output/'online_state_timeline.png',dpi=180);plt.close(figure)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
