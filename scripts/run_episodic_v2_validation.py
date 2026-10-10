"""Frozen v2 rehearsal and one-time Bearing3_4 operational validation."""
import argparse
import csv
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from state_interpreter.episodic_validation import evaluate_event_stream
from state_interpreter.relative_event_state import EpisodeTermination

CONFIG_ID='episodic_v2_bearing3_4_validation_20261010'
PARAMETERS={'calibration_steps':15,'sigma_multiplier':2.0,'enter_steps':5,'exit_steps':5,'trend_window':10,'min_reference_std':1e-8}
REHEARSAL_TOKEN='REHEARSE_EPISODIC_V2_ON_DEVELOPMENT_ARCHIVE'
VALIDATION_TOKEN='EXECUTE_EPISODIC_V2_BEARING3_4_ONCE'
PIN_PATHS=('src/state_interpreter/episodic_weak_event.py','src/state_interpreter/relative_event_state.py',
           'src/state_interpreter/episodic_validation.py','src/state_interpreter/features/bearing_features.py',
           'src/state_interpreter/adapters/xjtu_sy.py','scripts/run_episodic_v2_validation.py',
           'results/2026-10-09_relative_event_state_interface/states.csv',
           'results/2026-10-09_relative_event_state_interface/episode_terminations.csv')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def save_csv(path,rows,columns):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)

def contract(mode,bearing,token,dataset_root):
    if mode not in ('check','rehearsal','validation'):raise ValueError('unknown mode')
    if bearing!='Bearing3_4':raise PermissionError('only Bearing3_4 is a validation target')
    if mode=='rehearsal' and token!=REHEARSAL_TOKEN:raise PermissionError('rehearsal token required')
    if mode=='validation' and token!=VALIDATION_TOKEN:raise PermissionError('validation token required')
    if mode!='validation' and dataset_root is not None:raise ValueError('dataset root allowed only for validation')
    if mode=='validation' and dataset_root is None:raise ValueError('validation needs dataset root')

def expected_config():
    return {'experiment_id':CONFIG_ID,'condition':'40Hz10kN','validation_bearing':'Bearing3_4',
            'protected_blind':'Bearing3_5','parameters':PARAMETERS,'dependencies':{p:sha(ROOT/p) for p in PIN_PATHS},
            'input_method':'fixed 65-feature extractor; horizontal/vertical pair means for RMS and kurtosis',
            'rehearsal_output':'results/2026-10-10_episodic_v2_validation_rehearsal',
            'validation_output':'results/2026-10-10_episodic_v2_bearing3_4_validation',
            'records':'records/episodic_v2_bearing3_4_validation_20261010',
            'tokens':{'rehearsal':REHEARSAL_TOKEN,'validation':VALIDATION_TOKEN},
            'checks':['input_valid','references_valid','finite_present_outputs','causal_prefix_equality',
                      'entry_exit_rules_valid','termination_metadata_complete'],
            'interpretation':'operational vibration-change events, no physical fault accuracy; not project-wide blind test',
            'no_automatic_blind_release':True}

def verify_protocol(path):
    config=json.loads(path.read_text(encoding='utf-8'))
    lock=json.loads(path.with_suffix('.lock.json').read_text(encoding='utf-8'))
    if lock!={'config_sha256':sha(path),'experiment_id':CONFIG_ID} or config!=expected_config():
        raise ValueError('frozen protocol/dependency mismatch')
    return config

def run():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--mode',default='check');p.add_argument('--bearing',default='Bearing3_4')
    p.add_argument('--confirm');p.add_argument('--dataset-root',type=Path);a=p.parse_args()
    if a.freeze:
        if a.mode!='check' or a.dataset_root is not None:raise ValueError('freeze is metadata only')
        config=expected_config();save_json(a.freeze,config)
        save_json(a.freeze.with_suffix('.lock.json'),{'config_sha256':sha(a.freeze),'experiment_id':CONFIG_ID})
        print('Protocol frozen; no target data read.');return
    contract(a.mode,a.bearing,a.confirm,a.dataset_root)
    config=verify_protocol(a.config)
    record_dir=ROOT/config['records']
    if a.mode=='check':
        print('Preflight passed: frozen artifacts verified; no raw target data read.');return
    out=ROOT/config[a.mode+'_output']
    start_record=record_dir/(a.mode+'_started.json');completion=record_dir/(a.mode+'_completed.json')
    if out.exists() or start_record.exists() or completion.exists():raise FileExistsError('one-time output/record exists')
    if a.mode=='validation':
        rehearsal=json.loads((record_dir/'rehearsal_completed.json').read_text(encoding='utf-8'))
        if rehearsal['config_sha256']!=sha(a.config) or not rehearsal['all_checks_passed'] or not rehearsal['archive_exact_match']:
            raise ValueError('completed matching rehearsal required')
    record_dir.mkdir(parents=True,exist_ok=True)
    save_json(start_record,{'status':'started','config_sha256':sha(a.config),'mode':a.mode})
    out.mkdir()
    state_rows,terminations,summaries,checks_all=[],[],[],[]
    raw_manifest=[];exact=None
    if a.mode=='rehearsal':
        with (ROOT/PIN_PATHS[-2]).open(encoding='utf-8') as f:archive=list(csv.DictReader(f))
        for bearing in ('Bearing3_1','Bearing3_2','Bearing3_3'):
            rms=[r for r in archive if r['bearing_id']==bearing and r['provenance']=='rms_episodic_v2']
            kurt=[r for r in archive if r['bearing_id']==bearing and r['provenance']=='kurtosis_episodic_v2']
            if [r['step_id'] for r in rms]!=[r['step_id'] for r in kurt]:raise ValueError('unaligned archived features')
            steps=np.array([int(r['step_id']) for r in rms])
            pairs=np.array([[float(r['value']),float(k['value'])] for r,k in zip(rms,kurt)])
            rows,ends,sums,checks=evaluate_event_stream(bearing,steps,pairs)
            state_rows.extend(rows);terminations.extend(ends);summaries.extend(sums);checks_all.append(checks)
        exact=True
        if len(state_rows)!=len(archive):raise ValueError('archive row count mismatch')
        for row,old in zip(state_rows,archive):
            for key,value in row.items():
                if key in ('termination_now','last_termination'):
                    assert value==(json.loads(old[key+'_json']) if old[key+'_json'] else None)
                elif value is None:assert old[key]==''
                elif isinstance(value,bool):assert old[key]==str(value)
                elif isinstance(value,(int,float)):assert float(old[key])==float(value)
                else:assert old[key]==str(value)
        with (ROOT/PIN_PATHS[-1]).open(encoding='utf-8') as f:old_ends=list(csv.DictReader(f))
        assert len(old_ends)==len(terminations)
        for row,old in zip(terminations,old_ends):
            for key,value in row.items():assert old[key]==str(value)
    else:
        # Directly resolve only the validation directory: no sibling bearing enumeration.
        from state_interpreter.adapters.xjtu_sy import XJTUSYDatasetAdapter
        from state_interpreter.features.bearing_features import FEATURE_NAMES,XJTUBearingFeatureExtractor
        target=(a.dataset_root.resolve()/'40Hz10kN'/'Bearing3_4').resolve()
        if not target.is_relative_to(a.dataset_root.resolve()):raise ValueError('target escapes dataset root')
        files=XJTUSYDatasetAdapter._ordered_csv_files(target)
        reader=XJTUSYDatasetAdapter(target)
        extractor=XJTUBearingFeatureExtractor(condition='40Hz10kN')
        pair_rows=[];steps=[]
        for i,path in enumerate(files):
            if path.resolve().parent!=target:raise ValueError('measurement escapes target')
            raw_manifest.append({'measurement':path.name,'sha256':sha(path)})
            features=extractor.extract(reader._read_vibration(path).numpy().T)
            pair_rows.append([(float(features[FEATURE_NAMES.index('h_'+name)])+float(features[FEATURE_NAMES.index('v_'+name)]))/2 for name in ('rms','kurtosis')])
            steps.append(int(path.stem)-1)
            if (i+1)%100==0:print(f'Bearing3_4: extracted {i+1} measurements',flush=True)
        rows,ends,sums,checks=evaluate_event_stream('Bearing3_4',np.array(steps),np.array(pair_rows))
        state_rows.extend(rows);terminations.extend(ends);summaries.extend(sums);checks_all.append(checks)
    for row in state_rows:
        for key in ('termination_now','last_termination'):
            value = row.pop(key)
            row[key+'_json']=json.dumps(value) if value is not None else None
    save_csv(out/'states.csv',state_rows,list(state_rows[0]))
    end_columns=['bearing_id','feature']+[f.name for f in fields(EpisodeTermination)]
    save_csv(out/'episode_terminations.csv',terminations,end_columns)
    save_csv(out/'summary.csv',summaries,list(summaries[0]))
    if raw_manifest:save_json(out/'raw_input_manifest.json',raw_manifest)
    report={'status':'completed','mode':a.mode,'config_sha256':sha(a.config),'all_checks_passed':all(all(c.values()) for c in checks_all),
            'checks':checks_all,'archive_exact_match':exact,'state_rows':len(state_rows),
            'episode_count':len(terminations),'summaries':summaries,'protected_blind_not_read':'Bearing3_5',
            'validation_data_read':a.mode=='validation','interpretation':config['interpretation']}
    save_json(out/'report.json',report)
    save_json(completion,{**report,'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
