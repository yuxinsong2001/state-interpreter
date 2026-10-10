"""Compare frozen re-extraction with original archived held-out metrics."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
root,out=map(Path,sys.argv[1:3]);sys.path.insert(0,str(root/'src'))
from state_interpreter.signed_axis import signed_level
table=pd.read_csv(out/'heldout_latents.csv');metrics=pd.read_csv(out/'metrics.csv')
assert len(metrics)==9 and len(table)==16215
assert set(table.test_bearing)=={'Bearing3_1','Bearing3_2','Bearing3_3'}
assert np.isfinite(table.select_dtypes('number')).all().all()
checks=[]
for r in metrics.to_dict('records'):
    base=r['base_seed'];b=r['test_bearing']
    directory=root/f'results/2026-09-18_condition3_arm_b_encoder_lobo_seed_{base}'
    cp=torch.load(directory/f'{b}_checkpoint.pt',map_location='cpu',weights_only=False)
    archive=pd.read_csv(directory/'lobo_summary.csv').set_index('test_bearing').loc[b]
    g=table[(table.base_seed==base)&(table.test_bearing==b)].sort_values('step_id')
    z=g[[f'z_{i}' for i in range(8)]].to_numpy()
    level=signed_level(z,cp['axis'],calibration_steps=15,temporal_window=10)
    mask=g.step_id.to_numpy()>=25
    rho=float(spearmanr(g.step_id.to_numpy()[mask],level[mask]).statistic)
    mse_diff=abs(r['mean_reconstruction_mse']-float(archive.heldout_reconstruction_mse))
    rho_diff=abs(rho-float(archive.level_spearman))
    assert mse_diff<1e-5 and rho_diff<1e-5,(mse_diff,rho_diff)
    checks.append(dict(base_seed=base,test_bearing=b,archived_signed_rho=float(archive.level_spearman),recomputed_signed_rho=rho,absolute_rho_difference=rho_diff,absolute_mse_difference=mse_diff))
sources=['src/state_interpreter/preprocessing/stft.py','src/state_interpreter/encoders/autoencoder.py','src/state_interpreter/data/stft_dataset.py','src/state_interpreter/signed_axis.py']
report=dict(status='passed',checks=checks,source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},limits='Verification of forward extraction and archived metrics, not physical health semantics')
with (out/'verification.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
