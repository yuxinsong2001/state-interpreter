"""Descriptive post-validation audit; never changes event parameters."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
source=Path(sys.argv[1]);out=Path(sys.argv[2])
if out.exists():raise FileExistsError(out)
files=[source/'states.csv',source/'episode_terminations.csv',source/'report.json']
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with files[0].open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
with files[1].open(encoding='utf-8') as f:events=list(csv.DictReader(f))
assert len(rows)==3030 and len(events)==2
windows=[('calibration',0,15),('immediately_after',15,30),('early_30_100',30,100),('middle_100_1000',100,1000),('pre_event_1000_1417',1000,1417),('candidate_1417_1422',1417,1422),('confirmed_tail_1421_1515',1421,1515)]
stats=[];details=[]
for feature in ('rms','kurtosis'):
    data=[r for r in rows if r['provenance']==feature+'_episodic_v2']
    assert [int(r['step_id']) for r in data]==list(range(1515))
    y=np.array([float(r['value']) for r in data]);mu=y[:15].mean();sd=y[:15].std()
    threshold=mu+2*sd;z=(y-mu)/sd
    ready=data[15]
    assert mu==float(ready['reference_mean']) and sd==float(ready['reference_std'])
    for name,a,b in windows:
        v=y[a:b];zz=z[a:b]
        stats.append(dict(feature=feature,window=name,first_step=a,last_step=b-1,n=len(v),mean=float(v.mean()),median=float(np.median(v)),std=float(v.std()),min=float(v.min()),max=float(v.max()),median_reference_z=float(np.median(zz)),fraction_below_reference_minus_2sd=float(np.mean(zz < -2)),fraction_above_frozen_threshold=float(np.mean(v>threshold))))
    # Count above-threshold runs: fixed rules unchanged; no alternative event detector.
    runs=[];start=None
    for i in range(15,len(y)+1):
        above=i<len(y) and y[i]>threshold
        if above and start is None:start=i
        if not above and start is not None:
            runs.append({'start_step':start,'last_step':i-1,'length':i-start});start=None
    end=next(e for e in events if e['feature']==feature)
    assert int(end['candidate_start_step'])==1417 and int(end['confirm_step'])==1421
    assert np.all(y[1417:1422]>threshold)
    assert np.all(y[1421:]>threshold)
    details.append({'feature':feature,'above_threshold_runs':runs,'tail_all_above_threshold':True,'tail_min_reference_z':float(z[1421:].min()),'tail_median_reference_z':float(np.median(z[1421:])), 'post_calibration_fraction_below_minus_2sd':float(np.mean(z[15:1417]<-2)),'first5_mean':float(y[:5].mean()),'last5_of_calibration_mean':float(y[10:15].mean())})
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert before==after
out.mkdir()
with (out/'window_statistics.csv').open('x',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
report={'verification_status':'ANALYZED','method':'post-validation descriptive windows, no parameter tuning or raw-data replay','input_sha256':before,'input_unchanged':True,'validation_not_reexecuted':True,'protected_bearing3_5_not_read':True,'window_selection':'exploratory; calibration, adjacent windows, broad temporal regions and archived event boundaries; not confirmatory inference','features':details}
with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(report,indent=2));print(json.dumps(stats,indent=2))
