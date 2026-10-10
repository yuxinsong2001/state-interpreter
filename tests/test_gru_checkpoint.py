import pytest
import torch
from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
from state_interpreter.gru_checkpoint import load_online_gru_checkpoint

def fixture():
    model=GRUStateReadoutInterpreter(embedding_dim=8)
    return dict(model_state_dict=model.state_dict(),config=dict(calibration_steps=15,temporal_window=10,score_start=25),train_mean=torch.zeros(8),train_std=torch.ones(8),level_target_scale=torch.tensor(2.))

def test_load_and_independent_reset(tmp_path):
    path=tmp_path/'checkpoint.pt';torch.save(fixture(),path)
    a,b=load_online_gru_checkpoint(path),load_online_gru_checkpoint(path)
    assert a.update(torch.zeros(8)).step==0
    assert a.update(torch.zeros(8)).step==1
    assert b.update(torch.zeros(8)).step==0

@pytest.mark.parametrize('kind',['missing','window','level_only','nonfinite','zero_std'])
def test_reject(tmp_path,kind):
    value=fixture()
    if kind=='missing':del value['train_mean']
    elif kind=='window':value['config']['temporal_window']=20
    elif kind=='level_only':value['config']['level_only']=True
    elif kind=='zero_std':value['train_std']=torch.zeros(8)
    else:value['model_state_dict']['core.gru.weight_ih_l0'].fill_(float('nan'))
    path=tmp_path/'bad.pt';torch.save(value,path)
    with pytest.raises(ValueError):load_online_gru_checkpoint(path)
