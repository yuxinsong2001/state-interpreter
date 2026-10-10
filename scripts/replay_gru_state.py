"""Explicitly selected embedding CSV episodes -> relative states and plot."""
import argparse,csv,hashlib,json,math,sys
from dataclasses import asdict
from pathlib import Path
import torch

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--embeddings',type=Path,required=True)
    parser.add_argument('--bearing',action='append',required=True,help='Explicit episode IDs; repeat to select more')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acknowledge-matching-encoder',action='store_true',help='Confirm input uses training encoder coordinates, not a different z8 encoder')
    args=parser.parse_args()
    if not args.acknowledge_matching_encoder:parser.error('Confirm matching encoder coordinates; equal dimension alone is insufficient')
    if args.output.exists():parser.error('Output must not exist; old results are never overwritten')
    if len(set(args.bearing))!=len(args.bearing):parser.error('Duplicate episode selections')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from state_interpreter.gru_checkpoint import load_online_gru_checkpoint
    online=load_online_gru_checkpoint(args.checkpoint)
    d=online._model.embedding_dim
    rows={name:[] for name in args.bearing}
    with args.embeddings.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        needed={'bearing_id','step_id',*(f'z_{i}' for i in range(d))}
        if not needed.issubset(reader.fieldnames or []):raise ValueError('CSV columns missing')
        for row in reader:
            if row['bearing_id'] not in rows:continue
            step=int(row['step_id'])
            values=[float(row[f'z_{i}']) for i in range(d)]
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite embedding')
            rows[row['bearing_id']].append((step,values))
    output=[]
    for name,episode in rows.items():
        if len(episode)<26 or [step for step,_ in episode]!=list(range(len(episode))):
            raise ValueError('Each selected episode must be ordered, contiguous from zero, and have >=26 points')
        online.reset()
        for _,values in episode:
            record=asdict(online.update(torch.tensor(values,dtype=torch.float32)))
            record['bearing']=name;output.append(record)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for name in args.bearing:
        ready=[r for r in output if r['bearing']==name and r['phase']=='READY']
        for axis,field in zip(axes,['level','trend','movement']):axis.plot([r['step'] for r in ready],[r[field] for r in ready],label=name)
    for axis,label in zip(axes,['Level (relative proxy units)','Trend (proxy units / measurement)','Movement (standardized embedding RMSE)']):
        axis.set_ylabel(label);axis.grid(alpha=.25);axis.legend()
    axes[-1].set_xlabel('Measurement index (not seconds)')
    figure.suptitle('GRU relative state: selected embedding replay')
    figure.text(.01,.01,'Source: supplied embedding CSV and checkpoint; not physical health or live-sensor validation.',fontsize=9)
    figure.tight_layout(rect=(0,.03,1,.96))
    args.output.mkdir(parents=True)
    with (args.output/'states.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    figure.savefig(args.output/'state_timeline.png',dpi=180);plt.close(figure)
    report=dict(status='completed',checkpoint_sha256=digest(args.checkpoint),input_sha256=digest(args.embeddings),runner_sha256=digest(Path(__file__)),selected_episodes=args.bearing,total_rows=len(output),ready_rows=sum(r['phase']=='READY' for r in output),matching_encoder_acknowledged=True,matching_encoder_automatically_verified=False,limitations=['relative proxy, known Level bias retained','no health threshold or maintenance decision','unselected rows scanned but z not materialized','embedding replay, not raw-sensor deployment'])
    (args.output/'replay_record.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
