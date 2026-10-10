import pytest
import torch
from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
from state_interpreter.online_gru_state import OnlineGRUStateInterpreter


def setup():
    torch.manual_seed(42)
    model=GRUStateReadoutInterpreter(embedding_dim=8)
    with torch.no_grad(): model.core.prediction_head.weight.normal_(std=.1)
    raw=torch.randn(45,8)
    mean=torch.randn(8);std=torch.rand(8)+.5;scale=torch.tensor(3.)
    return model,raw,mean,std,scale


def test_offline_match_and_phases():
    model,raw,mean,std,scale=setup()
    online=OnlineGRUStateInterpreter(model,mean,std,scale)
    outputs=[online.update(z) for z in raw]
    assert all(r.level is None for r in outputs[:25])
    assert outputs[13].phase=='CALIBRATING' and outputs[14].phase=='WARMING'
    assert outputs[25].phase=='READY'
    z=(raw-raw[:15].mean(0)-mean)/std
    with torch.no_grad(): offline=model(z[None])
    levels=offline.scaled_relative_level[0,:,0]*scale
    for t in range(25,len(raw)):
        x=torch.arange(10,dtype=raw.dtype);x-=x.mean()
        segment=levels[t-9:t+1]
        trend=float((x*(segment-segment.mean())).sum()/x.square().sum())
        movement=float((z[t]-offline.next_embedding_prediction[0,t-1]).square().mean().sqrt())
        assert outputs[t].level==pytest.approx(float(levels[t]),abs=1e-5)
        assert outputs[t].trend==pytest.approx(trend,abs=1e-5)
        assert outputs[t].movement==pytest.approx(movement,abs=1e-5)
    assert model.training


def test_reset_replay_identical():
    model,raw,mean,std,scale=setup()
    online=OnlineGRUStateInterpreter(model,mean,std,scale)
    first=[online.update(z) for z in raw]
    online.reset()
    assert first==[online.update(z) for z in raw]


def test_invalid_does_not_advance():
    model,raw,mean,std,scale=setup()
    online=OnlineGRUStateInterpreter(model,mean,std,scale)
    for bad in [torch.ones(7),torch.full((8,),float('nan')),torch.ones(8,dtype=torch.float64)]:
        with pytest.raises(ValueError): online.update(bad)
    assert online.update(raw[0]).step==0


def test_future_not_used():
    model,raw,mean,std,scale=setup()
    a=OnlineGRUStateInterpreter(model,mean,std,scale)
    b=OnlineGRUStateInterpreter(model,mean,std,scale)
    changed=raw.clone();changed[35:]*=100
    assert [a.update(z) for z in raw[:35]]==[b.update(z) for z in changed[:35]]


def test_bad_statistics():
    model,raw,mean,std,scale=setup()
    with pytest.raises(ValueError): OnlineGRUStateInterpreter(model,mean,torch.zeros(8),scale)
    with pytest.raises(ValueError): OnlineGRUStateInterpreter(model,mean,std,float('nan'))
