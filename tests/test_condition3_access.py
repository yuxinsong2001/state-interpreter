import json, pytest
from pathlib import Path
from state_interpreter.condition3_access import Condition3AccessPolicy

def cfg():
 return {'status':'preregistered_no_bearing3_data_read','split':{'development':['Bearing3_1','Bearing3_2','Bearing3_3'],'validation':['Bearing3_4'],'blind':['Bearing3_5']},'tokens':{'development':'D','validation':'V','blind':'B'},'frozen_artifacts':[]}
def test_development_exact_only(tmp_path):
 p=Condition3AccessPolicy(cfg(),tmp_path); assert p.authorize('development',['Bearing3_1','Bearing3_2','Bearing3_3'],'D')==('Bearing3_1','Bearing3_2','Bearing3_3')
 with pytest.raises(ValueError): p.authorize('development',['Bearing3_1','Bearing3_4'],'D')
def test_wrong_token_fails(tmp_path):
 with pytest.raises(PermissionError): Condition3AccessPolicy(cfg(),tmp_path).authorize('development',['Bearing3_1','Bearing3_2','Bearing3_3'],'x')
def test_validation_requires_development(tmp_path):
 p=Condition3AccessPolicy(cfg(),tmp_path)
 with pytest.raises(PermissionError): p.authorize('validation',['Bearing3_4'],'V')
 assert p.authorize('validation',['Bearing3_4'],'V',development_record=True)==('Bearing3_4',)
def test_blind_is_one_time_and_locked(tmp_path):
 p=Condition3AccessPolicy(cfg(),tmp_path)
 with pytest.raises(PermissionError): p.authorize('blind',['Bearing3_5'],'B')
 assert p.authorize('blind',['Bearing3_5'],'B',validation_locked=True,blind_record=False)==('Bearing3_5',)
 with pytest.raises(PermissionError): p.authorize('blind',['Bearing3_5'],'B',validation_locked=True,blind_record=True)
def test_rejects_changed_split(tmp_path):
 c=cfg(); c['split']['validation']=['Bearing3_5']
 with pytest.raises(ValueError): Condition3AccessPolicy(c,tmp_path)
