import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/run_episodic_v2_validation.py'
spec=importlib.util.spec_from_file_location('event_validation_runner',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

@pytest.mark.parametrize('mode,bearing,token,root',[('validation','Bearing3_5',runner.VALIDATION_TOKEN,Path('/unused')),
 ('validation','Bearing3_4','wrong',Path('/unused')),('rehearsal','Bearing3_4','wrong',None),
 ('check','Bearing3_4',None,Path('/unused')),('validation','Bearing3_4',runner.VALIDATION_TOKEN,None)])
def test_access_rejected_before_data_read(mode,bearing,token,root):
    with pytest.raises((ValueError,PermissionError)):runner.contract(mode,bearing,token,root)

def test_matching_lock_and_tampered_config(tmp_path):
    p=tmp_path/'protocol.json'
    config=runner.expected_config();runner.save_json(p,config)
    runner.save_json(p.with_suffix('.lock.json'),{'config_sha256':runner.sha(p),'experiment_id':runner.CONFIG_ID})
    assert runner.verify_protocol(p)==config
    config['parameters']={**config['parameters'],'exit_steps':6}
    p.write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(ValueError):runner.verify_protocol(p)

def test_dependency_change_rejected_even_with_matching_lock(tmp_path):
    p=tmp_path/'protocol.json';config=runner.expected_config()
    key=next(iter(config['dependencies']));config['dependencies'][key]='0'*64
    runner.save_json(p,config)
    runner.save_json(p.with_suffix('.lock.json'),{'config_sha256':runner.sha(p),'experiment_id':runner.CONFIG_ID})
    with pytest.raises(ValueError):runner.verify_protocol(p)
