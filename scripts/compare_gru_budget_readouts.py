"""Compare paired GRU budgets; no extra training or threshold fitting."""
import sys,json
from pathlib import Path
import pandas as pd
import numpy as np
root,new=map(Path,sys.argv[1:3]);old=root/'results/2026-10-10_gru_state_v1_formal'
a=pd.read_csv(old/'metrics.csv');b=pd.read_csv(new/'metrics.csv')
m=a.merge(b,on=['seed','bearing'],suffixes=('_300','_600'),validate='one_to_one')
for col in ['forecast_mse','level_rmse']:
    m[col+'_relative_change']=(m[col+'_600']-m[col+'_300'])/m[col+'_300']
rows=[]
for tag,path in [('budget300',old),('budget600',new)]:
    df=pd.read_csv(path/'state_outputs.csv')
    for (seed,bearing),g in df.groupby(['seed','bearing']):
        g=g.sort_values('step')
        for i,part in enumerate(np.array_split(np.arange(len(g)),3)):
            h=g.iloc[part];error=h.level.to_numpy()-h.target_level.to_numpy()
            rows.append(dict(budget=tag,seed=int(seed),bearing=bearing,segment=['early','middle','late'][i],mean_signed_error=float(error.mean()),mae=float(abs(error).mean())))
assert not (new/'budget_comparison.csv').exists()
m.to_csv(new/'budget_comparison.csv',index=False);pd.DataFrame(rows).to_csv(new/'segment_bias_comparison.csv',index=False)
print(m.query("bearing=='Bearing1_4'")[['seed','best_epoch_600','stop_epoch_600','forecast_mse_300','forecast_mse_600','forecast_mse_relative_change','level_rmse_300','level_rmse_600','level_rmse_relative_change']].to_string(index=False))
print(pd.DataFrame(rows).query("bearing=='Bearing1_4'").to_string(index=False))
