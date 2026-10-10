"""Paired within-GRU optimization comparison, not architecture selection."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
root,new=map(Path,sys.argv[1:3]);old=root/'results/2026-10-10_gru_state_v1_budget600'
a=pd.read_csv(old/'metrics.csv');b=pd.read_csv(new/'metrics.csv')
m=a.merge(b,on=['seed','bearing'],suffixes=('_ordinary','_balanced'),validate='one_to_one')
for col in ['forecast_mse','level_rmse']:
    m[col+'_relative_change']=(m[col+'_balanced']-m[col+'_ordinary'])/m[col+'_ordinary']
rows=[]
for tag,path in [('ordinary',old),('balanced',new)]:
    df=pd.read_csv(path/'state_outputs.csv')
    for (seed,bearing),g in df.groupby(['seed','bearing']):
        g=g.sort_values('step')
        for i,part in enumerate(np.array_split(np.arange(len(g)),3)):
            h=g.iloc[part];error=h.level.to_numpy()-h.target_level.to_numpy()
            rows.append(dict(update=tag,seed=int(seed),bearing=bearing,segment=['early','middle','late'][i],mean_signed_error=float(error.mean()),mae=float(abs(error).mean())))
assert not (new/'optimization_comparison.csv').exists()
m.to_csv(new/'optimization_comparison.csv',index=False);pd.DataFrame(rows).to_csv(new/'segment_bias_comparison.csv',index=False)
history=pd.read_csv(new/'training_history.csv')
diag=history.groupby('seed').agg(epochs=('epoch','max'),fallback_count=('fallback','sum'),prediction_weight_min=('prediction_weight','min'),prediction_weight_max=('prediction_weight','max'),level_weight_min=('level_weight','min'),level_weight_max=('level_weight','max'),negative_cosine_fraction=('gradient_cosine',lambda x:float((x.dropna()<0).mean())))
diag.to_csv(new/'gradient_summary.csv')
print(m.query("bearing=='Bearing1_4'")[['seed','best_epoch_balanced','stop_epoch_balanced','forecast_mse_ordinary','forecast_mse_balanced','forecast_mse_relative_change','level_rmse_ordinary','level_rmse_balanced','level_rmse_relative_change']].to_string(index=False))
print(pd.DataFrame(rows).query("bearing=='Bearing1_4'").to_string(index=False));print(diag.to_string())
