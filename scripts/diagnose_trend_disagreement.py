"""Fixed descriptive amplitude strata; no noise-band tuning or gate changes."""
import sys,json,hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
root,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir(parents=True)
paths={'ae':root/'results/2026-10-10_relative_change_evaluation/state_traces.csv','ts2vec':root/'results/2026-10-10_ts2vec_relative_change_comparison/state_traces.csv'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={name:sha(p) for name,p in paths.items()};rows=[]
for model,p in paths.items():
    df=pd.read_csv(p);assert set(df.bearing)=={'Bearing3_1','Bearing3_2','Bearing3_3'}
    for b,g in df.groupby('bearing'):
        seeds=sorted(g.seed.unique())
        for a,c in itertools.combinations(seeds,2):
            x=g[g.seed==a].sort_values('step_id');y=g[g.seed==c].sort_values('step_id')
            assert np.array_equal(x.step_id,y.step_id)
            xx=x.trend.to_numpy();yy=y.trend.to_numpy()
            signs=lambda v:np.where(abs(v)<=1e-9,0,np.sign(v))
            disagree=signs(xx)!=signs(yy)
            # Within-trajectory magnitude percentile; retrospective, not a physical noise floor.
            rx=pd.Series(abs(xx)).rank(pct=True).to_numpy();ry=pd.Series(abs(yy)).rank(pct=True).to_numpy()
            masks={'all':np.ones(len(xx),bool),'both_low_quartile':np.maximum(rx,ry)<=.25,'both_lower_half':np.maximum(rx,ry)<=.5,'at_least_one_top_quartile':np.maximum(rx,ry)>=.75,'both_top_quartile':np.minimum(rx,ry)>=.75}
            for label,mask in masks.items():
                count=int(mask.sum());d=int(np.sum(disagree&mask))
                rows.append(dict(pipeline=model,bearing=b,seed_a=int(a),seed_b=int(c),stratum=label,n=count,disagreement_count=d,agreement=1-d/count if count else None,fraction_of_all_disagreements=d/int(disagree.sum()) if disagree.sum() else 0))
pd.DataFrame(rows).to_csv(out/'amplitude_strata.csv',index=False)
assert hashes=={name:sha(p) for name,p in paths.items()}
summary=pd.DataFrame(rows).groupby(['pipeline','bearing','stratum']).agg(min_agreement=('agreement','min'),max_agreement=('agreement','max'),min_disagreement_share=('fraction_of_all_disagreements','min'),max_disagreement_share=('fraction_of_all_disagreements','max'))
summary.to_csv(out/'summary.csv')
report=dict(status='completed',verification_status='ANALYZED',input_sha256=hashes,inputs_unchanged=True,no_training=True,no_gate_changes=True,definition='fixed per-seed absolute-Trend percentile strata, matched timestamp pairs',limitations=['relative amplitude strata do not identify sensor noise or practical significance','percentiles use whole archived scored trajectory; diagnostic not online calibration','strata overlap; shares cannot be summed','three seed pairs share seeds and devices; no independence or significance claim'])
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(summary.to_string())
