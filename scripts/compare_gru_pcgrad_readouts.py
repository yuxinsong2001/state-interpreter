"""Paired development comparison of updates within the same GRU."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root,new = map(Path,sys.argv[1:3])
old = root/'results/2026-10-10_gru_state_v1_budget600'
assert not (new/'optimization_comparison.csv').exists(), 'Refuse overwrite'
a,b = pd.read_csv(old/'metrics.csv'),pd.read_csv(new/'metrics.csv')
m = a.merge(b,on=['seed','bearing'],suffixes=('_ordinary','_pcgrad'),validate='one_to_one')
assert len(m)==12
for col in ('forecast_mse','level_rmse'):
    m[col+'_relative_change'] = m[col+'_pcgrad']/m[col+'_ordinary']-1
m.to_csv(new/'optimization_comparison.csv',index=False)
rows = []
for tag,path in [('ordinary',old),('pcgrad',new)]:
    df = pd.read_csv(path/'state_outputs.csv')
    for (seed,bearing),g in df.groupby(['seed','bearing']):
        g = g.sort_values('step')
        for i,indices in enumerate(np.array_split(np.arange(len(g)),3)):
            h = g.iloc[indices]
            error = h.level.to_numpy()-h.target_level.to_numpy()
            rows.append(dict(update=tag,seed=int(seed),bearing=bearing,segment=['early','middle','late'][i],mean_signed_error=float(error.mean()),mae=float(abs(error).mean())))
pd.DataFrame(rows).to_csv(new/'segment_bias_comparison.csv',index=False)
h = pd.read_csv(new/'training_history.csv')
diag = h.groupby('seed').agg(epochs=('epoch','max'),projection_count=('projected','sum'),zero_norm_count=('zero_norm_skipped','sum'),negative_cosine_fraction=('gradient_cosine',lambda x:float((x.dropna()<0).mean())),combined_norm_min=('combined_gradient_norm','min'),combined_norm_max=('combined_gradient_norm','max'))
diag.to_csv(new/'gradient_summary.csv')
print(m.query("bearing=='Bearing1_4'")[[ 'seed','best_epoch_pcgrad','stop_epoch_pcgrad','forecast_mse_ordinary','forecast_mse_pcgrad','forecast_mse_relative_change','level_rmse_ordinary','level_rmse_pcgrad','level_rmse_relative_change']].to_string(index=False))
print(pd.DataFrame(rows).query("bearing=='Bearing1_4'").to_string(index=False))
print(diag.to_string())
