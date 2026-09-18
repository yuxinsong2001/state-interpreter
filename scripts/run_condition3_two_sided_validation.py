from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.condition3_access import Condition3AccessPolicy
from state_interpreter.data import materialize_stft_data
from state_interpreter.signed_axis import fit_signed_axis,signed_level
from run_direct_generalization_v2 import load_frozen_model
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--config',required=True);a.add_argument('--access-config',required=True);a.add_argument('--confirm',required=True);x=a.parse_args()
 cpath=Path(x.config).resolve();c=json.loads(cpath.read_text(encoding='utf-8')); access=Condition3AccessPolicy.load(Path(x.access_config),ROOT); access.verify_artifacts()
 record_dir=ROOT/access.config['records_directory']; dev_record=record_dir/'development_completed.json'; locked_record=record_dir/'validation_locked.json'
 access.authorize('validation',[c['validation_bearing']],x.confirm,development_record=dev_record.is_file())
 if c['status']!='locked_before_validation_read' or c['blind_bearing']!='Bearing3_5': raise ValueError('validation lock changed')
 if h(ROOT/'runs/ae_baseline_z8_20260801/best_checkpoint.pt')!=c['frozen_hashes']['autoencoder_checkpoint'] or h(ROOT/'results/2026-09-18_condition3_model_localization_summary/experiment_report.json')!=c['frozen_hashes']['model_localization_report']: raise ValueError('frozen hash mismatch')
 out=ROOT/c['output_directory'];
 if out.exists() or locked_record.exists(): raise FileExistsError('validation may run only once')
 _,ae,prep,std=load_frozen_model(ROOT/'runs/ae_baseline_z8_20260801/best_checkpoint.pt'); seq={};meta={};started=time.time()
 for b in c['development_bearings']+[c['validation_bearing']]:
  data=materialize_stft_data(XJTUSYDatasetAdapter(Path(x.root),conditions=[c['condition']],bearings=[b]),prep); values=std.transform(data.tensors); zs=[]
  with torch.inference_mode():
   for i in range(0,len(values),32): zs.append(ae.encode(values[i:i+32]).cpu())
  seq[b]=torch.cat(zs).numpy(); steps=np.asarray(data.step_ids,int); meta[b]=(steps,steps/max(steps))
 axis=fit_signed_axis([seq[b] for b in c['development_bearings']],calibration_steps=15); test=c['validation_bearing']; signed=signed_level(seq[test],axis,calibration_steps=15,temporal_window=10); two=np.abs(signed);steps,life=meta[test];mask=(steps>=c['score_start_step'])&np.isfinite(signed)
 sr=float(spearmanr(life[mask],signed[mask]).statistic);tr=float(spearmanr(life[mask],two[mask]).statistic)
 summary={'bearing_id':test,'samples':int(len(steps)),'score_samples':int(mask.sum()),'signed_rho':sr,'two_sided_rho':tr,'two_sided_positive':bool(tr>0),'first_last_delta':float(two[mask][-1]-two[mask][0]),'backward_step_fraction':float(np.mean(np.diff(two[mask])<0)),'protected_blind_not_read':'Bearing3_5'}
 out.mkdir(parents=True);pd.DataFrame({'step_id':steps,'normalized_lifetime':life,'signed_level':signed,'two_sided_level':two}).to_csv(out/'validation_trajectory.csv',index=False);(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 report={'status':'completed_locked','config_sha256':h(cpath),'elapsed_seconds':time.time()-started,'summary':summary,'interpretation':'Validation of a formula locked after Condition 3 development; Bearing3_5 remains unread.'};(out/'experiment_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');record_dir.mkdir(parents=True,exist_ok=True);locked_record.write_text(json.dumps(report,indent=2),encoding='utf-8')
 fig,ax=plt.subplots(figsize=(10,5));ax.plot(life,signed,label='signed');ax.plot(life,two,label='two-sided');ax.grid(alpha=.3);ax.legend();ax.set(xlabel='Normalized lifetime',ylabel='Level');fig.tight_layout();fig.savefig(out/'validation_levels.png',dpi=180);plt.close(fig);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
