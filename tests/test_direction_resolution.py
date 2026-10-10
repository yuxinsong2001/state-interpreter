import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'state-interpreter/src'))
import pytest
import torch
try:
    from state_interpreter.direction_resolution import DirectionResolutionPolicy, DirectionLabel, ResolutionAwareRelativeInterpreter
except ModuleNotFoundError:
    from direction_resolution import DirectionResolutionPolicy, DirectionLabel, ResolutionAwareRelativeInterpreter
from state_interpreter.relative_temporal import RelativeTemporalStateInterpreter

def policy(limit=.1):return DirectionResolutionPolicy(limit,'synthetic-test-only','synthetic')
def make(p=None):return ResolutionAwareRelativeInterpreter(embedding_dim=1,calibration_steps=3,temporal_window=3,representation_id='synthetic',policy=p)
def run(m,values):return [m.update(torch.tensor([v],dtype=torch.float64)) for v in values]

@pytest.mark.parametrize('limit',[0,-1,float('nan'),float('inf')])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):policy(limit)

def test_identity_mismatch():
    with pytest.raises(ValueError):ResolutionAwareRelativeInterpreter(embedding_dim=1,representation_id='other',policy=policy())

def test_evidence_required():
    with pytest.raises(ValueError):DirectionResolutionPolicy(.1,'','synthetic')

def test_uncalibrated_not_stable():
    out=run(make(),[0]*12)
    assert out[0:3]==[None]*3
    assert out[3].direction==DirectionLabel.WINDOW_WARMUP
    assert out[-1].direction==DirectionLabel.UNCALIBRATED

def test_stable_unresolved_not_health():
    assert run(make(policy()),[0]*12)[-1].direction==DirectionLabel.UNRESOLVED

def test_away():
    assert run(make(policy()),[0]*3+list(range(12)))[-1].direction==DirectionLabel.AWAY

def test_toward():
    assert run(make(policy()),[0]*3+list(range(12))+list(range(11,-1,-1)))[-1].direction==DirectionLabel.TOWARD

def test_continuous_unchanged():
    vals=[0]*3+list(range(12));wrapped=run(make(policy()),vals)
    base=RelativeTemporalStateInterpreter(embedding_dim=1,calibration_steps=3,temporal_window=3)
    for w,b in zip(wrapped,run(base,vals)):
        if w is not None:assert torch.equal(w.continuous.state,b.state)

def test_reset():
    m=make(policy());first=run(m,[0]*3+list(range(12)));m.reset();second=run(m,[0]*3+list(range(12)))
    assert [None if o is None else o.direction for o in first]==[None if o is None else o.direction for o in second]

def test_prefix_causal():
    values=[0]*3+list(range(12));a=run(make(policy()),values);b=run(make(policy()),values[:9])
    for x,y in zip(a,b):
        if x is not None:assert x.direction==y.direction and torch.equal(x.continuous.state,y.continuous.state)
