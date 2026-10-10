"""Fixed q80 early-operation engineering pilot; no threshold search."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
root,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir(parents=True)
trace=root/'results/2026-10-10_relative_change_evaluation/state_traces.csv'
cp=root/'results/2026-09-18_condition3_arm_b_encoder_lobo_seed_20260918/Bearing3_3_checkpoint.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={'trace':sha(trace),'checkpoint':sha(cp)}
df=pd.read_csv(trace);g=df[(df.bearing=='Bearing3_3')&(df.seed==20260918)].sort_values('step_id').copy()
fit=g[(g.step_id>=25)&(g.step_id<65)];assert len(fit)==40
limit=float(np.quantile(abs(fit.trend),.8,method='linear'))
mad=float(np.median(abs(fit.level-np.median(fit.level))))
shift=float(np.median(fit.level.iloc[-10:])-np.median(fit.level.iloc[:10]))
rho=float(spearmanr(fit.step_id,fit.level).statistic)
drift=bool(abs(rho)>=.8 and abs(shift)>3*mad)
valid=bool(np.isfinite(limit) and limit>0 and not drift)
g['hypothetical_direction']=np.where(abs(g.trend)<=limit,'within_resolution',np.where(g.trend>0,'moving_away','moving_toward'))
g['active_direction']=g.hypothetical_direction if valid else 'direction_uncalibrated'
g.loc[g.step_id<85,'active_direction']='direction_uncalibrated'
rows=[]
for name,start,end in [('fit',25,65),('reference_check',65,85),('evaluation',85,int(g.step_id.max())+1)]:
    h=g[(g.step_id>=start)&(g.step_id<end)];labels=h.hypothetical_direction.to_numpy()
    directional=labels!='within_resolution'
    # Count adjacent directional reversals only; unresolved rows are not compressed away.
    flips=int(np.sum((labels[1:]!=labels[:-1]) & directional[1:] & directional[:-1]))
    rows.append(dict(segment=name,start=start,end_exclusive=end,n=len(h),directional_coverage=float(np.mean(directional)),away_count=int(np.sum(labels=='moving_away')),toward_count=int(np.sum(labels=='moving_toward')),unresolved_count=int(np.sum(~directional)),adjacent_direction_reversals=flips,active_policy=valid and name=='evaluation'))
g.to_csv(out/'state_labels.csv',index=False);pd.DataFrame(rows).to_csv(out/'segment_metrics.csv',index=False)
report=dict(status='completed',verification_status='ANALYZED',quantile=.8,trend_limit=limit,reference_drift_suspected=drift,reference_level_time_rho=rho,reference_level_mad=mad,reference_first_last_median_shift=shift,policy_activated=valid,checkpoint_sha256=hashes['checkpoint'],inputs_unchanged=True,limitations=['single previously used development bearing','no certified stationary reference or health labels','non-overlapping segments remain temporally dependent; boundary smoothing carries past context','q80 is engineering choice, not validated false-alarm rate','coverage is not accuracy','drift flag is engineering guard, not statistical stationarity test'])
assert hashes=={'trace':sha(trace),'checkpoint':sha(cp)}
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(report,indent=2))
