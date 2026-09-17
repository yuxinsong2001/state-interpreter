import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def h(p):
 d=hashlib.sha256(); d.update(p.read_bytes()); return d.hexdigest()
def runs(steps):
 out=[]
 for s in sorted(steps):
  if not out or s>out[-1][-1]+1: out.append([s])
  else: out[-1].append(s)
 return out
def main():
 a=argparse.ArgumentParser(); a.add_argument('--config',required=True); cpath=Path(a.parse_args().config).resolve(); c=json.loads(cpath.read_text())
 statep=ROOT/c['state_trajectories']; errp=ROOT/c['per_seed_errors']; out=ROOT/c['output_directory']
 if out.exists(): raise FileExistsError(out)
 st=pd.read_csv(statep); er=pd.read_csv(errp); er=er[er.arm=='residual'].copy()
 tr=c['training_bearings']; ev=c['evaluation_bearings']; start=c['score_start_step']
 train=st[st.bearing_id.isin(tr)&(st.step_id>=start)]
 threshold=float(train.movement.quantile(c['movement_quantile']))
 trend_threshold=float(train.trend.abs().quantile(c['trend_absolute_quantile']))
 seed_thresholds={int(seed):float(np.sqrt(g.model_step_mse*8).quantile(c['movement_quantile'])) for seed,g in er[er.bearing_id.isin(tr)].groupby('seed')}
 rows=[]; points=[]
 for b in ev:
  g=st[(st.bearing_id==b)&(st.step_id>=start)].copy(); flagged=g[g.movement>=threshold]; rr=runs(flagged.step_id.tolist()); persistent=[r for r in rr if len(r)>=c['minimum_persistent_steps']]
  high=g[g.trend.abs()>=trend_threshold].step_id.to_numpy(int)
  agreements=[]
  for step in flagged.step_id:
   count=0
   for seed,t in seed_thresholds.items():
    q=er[(er.seed==seed)&(er.bearing_id==b)&(er.target_step==step)]
    if len(q) and float(np.sqrt(q.iloc[0].model_step_mse*8))>=t: count+=1
   agreements.append(count)
   points.append({'bearing_id':b,'step_id':int(step),'movement':float(g[g.step_id==step].iloc[0].movement),'seed_agreement':count,'distance_to_high_abs_trend':int(np.min(np.abs(high-step))) if len(high) else None})
  rows.append({'bearing_id':b,'threshold':threshold,'event_points':len(flagged),'event_fraction':len(flagged)/len(g),'persistent_runs':len(persistent),'max_run_length':max([len(r) for r in rr],default=0),'three_seed_agreement_points':sum(x==c['seed_count'] for x in agreements),'max_movement':float(g.movement.max()),'max_movement_step':int(g.loc[g.movement.idxmax(),'step_id'])})
 out.mkdir(parents=True); pd.DataFrame(rows).to_csv(out/'event_summary.csv',index=False); pd.DataFrame(points).to_csv(out/'event_points.csv',index=False)
 report={'experiment_id':c['experiment_id'],'ensemble_threshold':threshold,'trend_threshold':trend_threshold,'seed_thresholds':seed_thresholds,'config_sha256':h(cpath),'inputs':{str(statep.relative_to(ROOT)):h(statep),str(errp.relative_to(ROOT)):h(errp)},'interpretation':'Change candidates only; no fault labels.'}
 (out/'experiment_report.json').write_text(json.dumps(report,indent=2)); print(pd.DataFrame(rows).to_string(index=False)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
