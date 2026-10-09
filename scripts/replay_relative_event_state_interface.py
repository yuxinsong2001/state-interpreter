"""Verify the new interface against frozen per-step outputs; CSVs only."""
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'src'))
from state_interpreter.episodic_weak_event import EpisodeConfig
from state_interpreter.relative_event_state import RelativeEventStateInterpreter

source=root/'results/2026-10-09_condition3_episodic_v2_development'
out=root/'results/2026-10-09_relative_event_state_interface'
if out.exists():raise FileExistsError(out)
paths=[source/'trajectories.csv',source/'episodes.csv']
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with paths[0].open(encoding='utf-8') as f:archived=list(csv.DictReader(f))
with paths[1].open(encoding='utf-8') as f:episodes=list(csv.DictReader(f))
states,terminations,examples=[],[],[]
for bearing in ('Bearing3_1','Bearing3_2','Bearing3_3'):
    for kind in ('rms','kurtosis'):
        provenance=kind+'_episodic_v2'
        rows=[r for r in archived if r['bearing_id']==bearing and r['provenance']==provenance]
        d=RelativeEventStateInterpreter(EpisodeConfig(provenance=provenance),bearing_id=bearing)
        for row in rows:
            state=d.update(bearing_id=bearing,step_id=int(row['step_id']),value=float(row['value']))
            raw=asdict(state.observation);raw['phase']=state.observation.phase.value
            for key,value in raw.items():
                expected=row[key]
                if value is None:assert expected==''
                elif isinstance(value,bool):assert expected==str(value)
                elif isinstance(value,(int,float)):assert float(expected)==float(value)
                else:assert expected==str(value)
            flat={**raw,'schema_version':state.schema_version,
                  'feature_deviation_z':state.feature_deviation_z,
                  'feature_trend_z_per_measurement':state.feature_trend_z_per_measurement,
                  'trend_observation_count':state.trend_observation_count,
                  'transition':state.transition,
                  'termination_now_json':None if state.termination_now is None else json.dumps(asdict(state.termination_now)),
                  'last_termination_json':None if state.last_termination is None else json.dumps(asdict(state.last_termination))}
            states.append(flat)
            if state.termination_now:
                terminations.append({'bearing_id':bearing,'feature':kind,**asdict(state.termination_now)})
            if state.transition!='none' and bearing=='Bearing3_1' and kind=='rms':
                examples.append(state.to_dict())
        end=d.finish_record()
        if end:terminations.append({'bearing_id':bearing,'feature':kind,**asdict(end)})

assert len(states)==len(archived)==10810
assert len(terminations)==len(episodes)
for term in terminations:
    event=next(e for e in episodes if e['bearing_id']==term['bearing_id'] and e['feature']==term['feature'] and int(e['episode_id'])==term['episode_id'])
    assert term['observed_active_measurements']==int(event['active_measurements'])
    assert term['confirm_step']==int(event['confirm_step'])
    assert term['right_censored']==(event['right_censored_at_record_end']=='True')
    if event['exit_confirm_step']:
        assert term['reason']=='consecutive_non_exceedances'
        assert term['termination_observed_step']==int(event['exit_confirm_step'])
    else:assert term['reason']=='record_end'
out.mkdir()
for name,records in [('states.csv',states),('episode_terminations.csv',terminations)]:
    with (out/name).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
with (out/'example_state_messages.json').open('x',encoding='utf-8') as f:
    json.dump(examples,f,indent=2,allow_nan=False)
report={'status':'verified_against_archived_csv','state_rows':len(states),'termination_records':len(terminations),
        'completed_events':sum(not t['right_censored'] for t in terminations),
        'right_censored_events':sum(t['right_censored'] for t in terminations),
        'all_original_observation_fields_exactly_equal':True,'all_episode_metadata_matches_archive':True,
        'protected_not_read':['Bearing3_4','Bearing3_5'],'input_sha256':hashes,
        'module_sha256':hashlib.sha256((root/'src/state_interpreter/relative_event_state.py').read_bytes()).hexdigest(),
        'interpretation':'interface verification; no new health-performance evidence'}
with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
