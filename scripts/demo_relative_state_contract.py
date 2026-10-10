"""Synthetic interface demonstration, NOT a health or model benchmark."""
import argparse, csv, json, sys
from pathlib import Path
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo/'src'))
    from state_interpreter.direction_resolution import DirectionResolutionPolicy
    from state_interpreter.reference_qualified import ReferenceQualifiedInterpreter, ReferenceQualification, ReferenceStatus
    if args.output.exists(): raise FileExistsError('Use a new output directory; no overwrite')
    policy=DirectionResolutionPolicy(.05,'synthetic-demo-only','synthetic-z8-v1')
    accepted=ReferenceQualification(ReferenceStatus.ACCEPTED,'synthetic-episode','synthetic-z8-v1',15,10,
        'synthetic construction: first 15 vectors are exactly zero','Demonstrates explicit declaration only; not real qualification')
    common=dict(embedding_dim=8,episode_id='synthetic-episode',representation_id='synthetic-z8-v1',calibration_steps=15,temporal_window=10,policy=policy)
    unreviewed=ReferenceQualifiedInterpreter(**common)
    reviewed=ReferenceQualifiedInterpreter(**common,qualification=accepted)
    trajectory=[0.]*30+[i*.2 for i in range(1,31)]+[6.]*20+[6.-i*.2 for i in range(1,31)]+[0.]*20
    rows=[]
    for step,amplitude in enumerate(trajectory):
        z=torch.zeros(8,dtype=torch.float64);z[0]=amplitude
        a,b=unreviewed.update(z),reviewed.update(z)
        if a is None:
            assert b is None
            rows.append(dict(step=step,input_axis=amplitude,level='',trend='',movement='',unreviewed_direction='calibrating',synthetic_accepted_direction='calibrating'))
        else:
            assert torch.equal(a.continuous.state,b.continuous.state)
            assert a.direction.value=='direction_uncalibrated'
            rows.append(dict(step=step,input_axis=amplitude,level=float(a.continuous.level),trend=float(a.continuous.trend),movement=float(a.continuous.movement),unreviewed_direction=a.direction.value,synthetic_accepted_direction=b.direction.value))
    labels={r['synthetic_accepted_direction'] for r in rows}
    assert {'window_warmup','within_resolution','moving_away','moving_toward'}<=labels
    args.output.mkdir(parents=True)
    with (args.output/'demo_states.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    record=dict(status='completed',synthetic_only=True,steps=len(rows),embedding_dim=8,calibration_steps=15,temporal_window=10,synthetic_trend_limit=.05,continuous_outputs_identical=True,unreviewed_directions_suppressed=True,real_data_read=False,training=False,labels=sorted(labels))
    (args.output/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
