import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'state-interpreter/src'))
import pytest
import torch
try:
    from state_interpreter.reference_qualified import ReferenceStatus as S, ReferenceQualification as Q, ReferenceQualifiedInterpreter as I
except ModuleNotFoundError:
    from reference_qualified import ReferenceStatus as S, ReferenceQualification as Q, ReferenceQualifiedInterpreter as I
from state_interpreter.direction_resolution import DirectionResolutionPolicy as P, DirectionLabel as L

def make(q=None, p=True):
    return I(embedding_dim=1,episode_id='b1',representation_id='cp1',calibration_steps=3,
             temporal_window=3,policy=P(.1,'synthetic','cp1') if p else None,qualification=q)
def qualification(status=S.ACCEPTED, episode='b1'):
    return Q(status,episode,'cp1',3,3,'synthetic-only','test declaration, not real evidence')
def run(m):
    return [m.update(torch.tensor([float(x)])) for x in [0]*3+list(range(12))]

@pytest.mark.parametrize('q',[None,qualification(S.UNREVIEWED),qualification(S.REJECTED)])
def test_unqualified_suppressed(q):
    o=run(make(q))[-1]
    assert o.direction==L.UNCALIBRATED and o.suppression_reason
    assert o.continuous.state.isfinite().all()

def test_accepted():
    assert run(make(qualification()))[-1].direction==L.AWAY

def test_accepted_without_threshold():
    assert run(make(qualification(),False))[-1].direction==L.UNCALIBRATED

def test_scope():
    with pytest.raises(ValueError): make(qualification(episode='different'))

def test_reset_same_episode():
    m=make(); a=run(m);m.reset();b=run(m)
    for x,y in zip(a,b):
        if x is not None:
            assert x.direction==y.direction
            assert torch.equal(x.continuous.state,y.continuous.state)

def test_continuous_independent_of_qualification():
    for a,b in zip(run(make()),run(make(qualification()))):
        if a is not None: assert torch.equal(a.continuous.state,b.continuous.state)
