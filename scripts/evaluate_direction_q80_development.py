"""Apply the frozen pilot rule to archived development traces, without tuning."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root, out = map(Path, sys.argv[1:3])
assert not out.exists(), 'Refuse to overwrite results'
source = root/'results/2026-10-10_relative_change_evaluation/state_traces.csv'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_hash = sha(source)
df = pd.read_csv(source)
assert set(df.bearing) == {'Bearing3_1','Bearing3_2','Bearing3_3'}
assert set(df.seed) == {20260918,20260921,20260924}
rows, labels, hashes = [], [], {}
for (bearing, seed), raw in df.groupby(['bearing','seed']):
    g = raw.sort_values('step_id').copy()
    assert not g.step_id.duplicated().any()
    assert np.isfinite(g[['level','trend','movement']]).all().all()
    checkpoint = root/f'results/2026-09-18_condition3_arm_b_encoder_lobo_seed_{seed}/{bearing}_checkpoint.pt'
    hashes[str(checkpoint.relative_to(root))] = sha(checkpoint)
    fit = g[(g.step_id>=25)&(g.step_id<65)]
    assert len(fit)==40
    threshold = float(np.quantile(abs(fit.trend),.8,method='linear'))
    mad = float(np.median(abs(fit.level-np.median(fit.level))))
    shift = float(np.median(fit.level.iloc[-10:])-np.median(fit.level.iloc[:10]))
    rho = float(spearmanr(fit.step_id,fit.level).statistic)
    drift = bool(abs(rho)>=.8 and abs(shift)>3*mad)
    accepted = bool(np.isfinite(threshold) and threshold>0 and not drift)
    g['hypothetical_direction'] = np.where(abs(g.trend)<=threshold,'within_resolution',np.where(g.trend>0,'moving_away','moving_toward'))
    g['active_direction'] = np.where((g.step_id>=85)&accepted,g.hypothetical_direction,'direction_uncalibrated')
    g['trend_limit'] = threshold
    labels.append(g)
    for segment,start,end in [('fit',25,65),('reference_check',65,85),('evaluation',85,int(g.step_id.max())+1)]:
        h=g[(g.step_id>=start)&(g.step_id<end)]
        a=h.hypothetical_direction.to_numpy(); direction=a!='within_resolution'
        rows.append(dict(bearing=bearing,seed=int(seed),segment=segment,n=len(h),trend_limit=threshold,fit_rho=rho,fit_mad=mad,fit_shift=shift,drift_guard=drift,rule_accepted=accepted,hypothetical_coverage=float(direction.mean()),active_directional_coverage=float(direction.mean()) if accepted and segment=='evaluation' else 0.,away=int(sum(a=='moving_away')),toward=int(sum(a=='moving_toward')),unresolved=int(sum(~direction)),adjacent_reversals=int(np.sum((a[1:]!=a[:-1])&direction[1:]&direction[:-1]))))
metrics=pd.DataFrame(rows)
pilot=json.loads((root/'results/2026-10-10_direction_calibration_pilot/report.json').read_text())
reference=metrics[(metrics.bearing=='Bearing3_3')&(metrics.seed==20260918)&(metrics.segment=='evaluation')].iloc[0]
assert reference.trend_limit==pilot['trend_limit']
assert reference.away==32 and reference.unresolved==254
assert source_hash==sha(source)
assert all(sha(root/p)==h for p,h in hashes.items())
out.mkdir(parents=True)
metrics.to_csv(out/'segment_metrics.csv',index=False)
pd.concat(labels).to_csv(out/'state_labels.csv',index=False)
(out/'run_record.json').write_text(json.dumps(dict(status='completed',source_sha256=source_hash,checkpoint_hashes=hashes,quantile=.8,calibration_steps=15,temporal_window=10,fit_range=[25,65],reference_check_range=[65,85],evaluation_start=85,inputs_unchanged=True,pilot_reproduction_exact=True,raw_data_reads=0,protected_data_reads=0,training=False),indent=2),encoding='utf-8')
print(metrics[metrics.segment=='evaluation'].to_string(index=False))
