import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'state-interpreter/src'))
import pytest
import torch
from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
try:
    from state_interpreter.gru_task_balance import balanced_shared_gradients
except ModuleNotFoundError:
    from gru_task_balance import balanced_shared_gradients

def setup():
    torch.manual_seed(42);m=GRUStateReadoutInterpreter(embedding_dim=8)
    with torch.no_grad():m.core.prediction_head.weight.normal_(std=.1)
    z=torch.randn(1,12,8);o=m(z)
    return m,(o.next_embedding_prediction[:,:-1]-z[:,1:]).square().mean(),o.scaled_relative_level.square().mean()

def test_formula_heads_and_no_parameter_update():
    m,p,l=setup();shared=list(m.core.gru.parameters());ph=list(m.core.prediction_head.parameters());lh=list(m.level_head.parameters())
    gp=torch.autograd.grad(p,shared+ph,retain_graph=True);gl=torch.autograd.grad(l,shared+lh,retain_graph=True)
    before={n:v.clone() for n,v in m.state_dict().items()}
    d=balanced_shared_gradients(m,p,l);assert not d.fallback
    assert d.prediction_weight*d.prediction_norm==pytest.approx(d.level_weight*d.level_norm)
    for param,a,b in zip(shared,gp,gl):torch.testing.assert_close(param.grad,d.prediction_weight*a+d.level_weight*b)
    for param,g in zip(ph,gp[len(shared):]):torch.testing.assert_close(param.grad,g)
    for param,g in zip(lh,gl[len(shared):]):torch.testing.assert_close(param.grad,g)
    assert all(torch.equal(before[n],v) for n,v in m.state_dict().items())

def test_zero_gradient_fallback():
    m,p,l=setup();d=balanced_shared_gradients(m,p*0,l)
    assert d.fallback and d.prediction_weight==1 and d.level_weight==1
    assert all(torch.isfinite(x.grad).all() for x in m.parameters())

def test_optimizer_can_step():
    m,p,l=setup();opt=torch.optim.Adam(m.parameters(),lr=.001);opt.zero_grad()
    before=m.core.gru.weight_ih_l0.detach().clone()
    balanced_shared_gradients(m,p,l);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    assert not torch.equal(before,m.core.gru.weight_ih_l0)

def test_nonfinite_rejected_before_grads_assigned():
    m,p,l=setup()
    with pytest.raises(ValueError):balanced_shared_gradients(m,p*float('nan'),l)
    assert all(x.grad is None for x in m.parameters())

@pytest.mark.parametrize('eps',[0,-1,float('nan')])
def test_epsilon(eps):
    m,p,l=setup()
    with pytest.raises(ValueError):balanced_shared_gradients(m,p,l,epsilon=eps)
