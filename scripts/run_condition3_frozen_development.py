"""One-time frozen v2 development diagnostic on Bearing3_1..Bearing3_3."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch
from scipy.stats import spearmanr
from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.condition3_access import Condition3AccessPolicy
from state_interpreter.data import materialize_stft_data
from state_interpreter.relative_state_v2 import RelativeStateInterpreterV2
from state_interpreter.residual_gru import ResidualGRUStateInterpreter
from run_direct_generalization_v2 import load_frozen_model

def load_residuals(paths):
 models=[]; mean=std=None
 for p in paths:
  c=torch.load(p,weights_only=True,map_location='cpu'); m=ResidualGRUStateInterpreter(embedding_dim=c['embedding_dim'],hidden_dim=c['hidden_dim'],num_layers=c['num_layers']); m.load_state_dict(c['model_state_dict']); m.eval(); models.append(m)
  if mean is None: mean,std=c['feature_mean'],c['feature_std']
  elif not torch.equal(mean,c['feature_mean']) or not torch.equal(std,c['feature_std']): raise ValueError('residual normalization mismatch')
 return models,mean,std
def main():
 a=argparse.ArgumentParser(); a.add_argument('--root',required=True); a.add_argument('--config',required=True); a.add_argument('--confirm',required=True); a.add_argument('--output-dir',required=True); x=a.parse_args()
 cp=Path(x.config).resolve(); policy=Condition3AccessPolicy.load(cp,ROOT); policy.verify_artifacts(); bearings=policy.authorize('development',policy.config['split']['development'],x.confirm)
 out=Path(x.output_dir).resolve(); record_dir=ROOT/policy.config['records_directory']; record=record_dir/'development_completed.json'
 if out.exists() or record.exists(): raise FileExistsError('development output or completion record already exists')
 artifacts={i['path']:ROOT/i['path'] for i in policy.config['frozen_artifacts']}; _,ae,prep,standardizer=load_frozen_model(artifacts['runs/ae_baseline_z8_20260801/best_checkpoint.pt'])
 axis=np.asarray(json.loads(artifacts['results/2026-09-17_signed_axis_interpreter/axis.json'].read_text())['axis'])
 rpaths=[artifacts[f'results/2026-09-17_residual_gru_termination/residual_seed_{s}_checkpoint.pt'] for s in (20260916,20260917,20260918)]; models,mean,std=load_residuals(rpaths)
 movement_reference=0.05926334485411644; threshold=policy.config['frozen_parameters']['movement_threshold']; rows=[]; summaries=[]; started=time.time()
 for b in bearings:
  adapter=XJTUSYDatasetAdapter(Path(x.root),conditions=[policy.config['target_condition']],bearings=[b]); data=materialize_stft_data(adapter,prep); inputs=standardizer.transform(data.tensors)
  zs=[]; mses=[]
  with torch.inference_mode():
   for i in range(0,len(inputs),32):
    o=ae(inputs[i:i+32]); zs.append(o.z.cpu()); mses.append((o.reconstruction-inputs[i:i+32]).square().mean((1,2,3)).cpu())
  z=torch.cat(zs); mse=torch.cat(mses); interp=RelativeStateInterpreterV2(axis=axis,residual_models=models,feature_mean=mean,feature_std=std,movement_reference=movement_reference,calibration_steps=15,temporal_window=10); denom=max(data.step_ids); local=[]
  for i,v in enumerate(z):
   state=interp.update(v); step=int(data.step_ids[i]); life=step/max(1,denom)
   row={'bearing_id':b,'step_id':step,'normalized_lifetime':life,'reconstruction_mse':float(mse[i]),'level':'','trend':'','movement':''}
   if state is not None: row.update(level=state.level,trend=state.trend,movement=state.movement); local.append(row)
   rows.append(row)
  scored=[r for r in local if r['step_id']>=25]; level=np.array([r['level'] for r in scored]); life=np.array([r['normalized_lifetime'] for r in scored]); mov=np.array([r['movement'] for r in scored])
  summaries.append({'bearing_id':b,'samples':len(rows) if len(bearings)==1 else len(data.step_ids),'ready_scored':len(scored),'level_spearman':float(spearmanr(life,level).statistic),'level_first_last_delta':float(level[-1]-level[0]),'movement_event_fraction':float(np.mean(mov>=threshold)),'movement_median':float(np.median(mov)),'movement_max':float(np.max(mov)),'finite':bool(np.isfinite(np.c_[level,mov]).all()),'reconstruction_mse_mean':float(mse.mean())})
 out.mkdir(parents=True); 
 for name,data_rows in [('state_trajectories.csv',rows),('summary.csv',summaries)]:
  with (out/name).open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=data_rows[0].keys());w.writeheader();w.writerows(data_rows)
 fig,axes=plt.subplots(3,1,figsize=(11,10));
 for b in bearings:
  q=[r for r in rows if r['bearing_id']==b and r['level']!=''];
  for ax,k in zip(axes,('level','trend','movement')): ax.plot([r['normalized_lifetime'] for r in q],[r[k] for r in q],label=b);ax.grid(alpha=.3);ax.legend()
 fig.tight_layout();fig.savefig(out/'condition3_development.png',dpi=180);plt.close(fig)
 report={'status':'completed','bearings':list(bearings),'protected_bearings_not_read':['Bearing3_4','Bearing3_5'],'movement_threshold':threshold,'elapsed_seconds':time.time()-started,'summaries':summaries};(out/'report.json').write_text(json.dumps(report,indent=2));record_dir.mkdir(parents=True,exist_ok=True);record.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__': main()
