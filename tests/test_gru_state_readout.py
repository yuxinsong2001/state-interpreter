import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'state-interpreter/src'))
import importlib.util
import pytest
import torch
try:
    from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter
except ModuleNotFoundError:
    spec=importlib.util.spec_from_file_location('state_interpreter.gru_state_readout',Path(__file__).parent/'gru_state_readout.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    GRUStateReadoutInterpreter=module.GRUStateReadoutInterpreter

def model():
    torch.manual_seed(42)
    return GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8)

def test_shapes_nonnegative_and_initial_persistence():
    m=model();x=torch.randn(2,12,8);o=m(x)
    assert o.hidden_sequence.shape==(2,12,8)
    assert o.scaled_relative_level.shape==(2,12,1)
    assert o.final_hidden.shape==(1,2,8)
    assert torch.isfinite(o.scaled_relative_level).all() and (o.scaled_relative_level>=0).all()
    torch.testing.assert_close(o.next_embedding_prediction,x,atol=0,rtol=0)

def test_both_losses_have_expected_gradients():
    m=model();x=torch.randn(2,12,8);o=m(x)
    loss=(o.next_embedding_prediction[:,:-1]-x[:,1:]).square().mean()+o.scaled_relative_level.square().mean()
    loss.backward()
    for part in [m.core.gru,m.core.prediction_head,m.level_head]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in part.parameters())
        assert sum(float(p.grad.abs().sum()) for p in part.parameters())>0

def test_level_gradient_reaches_shared_gru():
    m=model();m(torch.randn(1,10,8)).scaled_relative_level.sum().backward()
    assert m.core.gru.weight_ih_l0.grad.abs().sum()>0

def test_prefix_causal_after_nonzero_prediction_head():
    m=model()
    with torch.no_grad():m.core.prediction_head.weight.normal_(std=.1)
    x=torch.randn(1,16,8);a=m(x);b=m(x[:,:7])
    torch.testing.assert_close(a.scaled_relative_level[:,:7],b.scaled_relative_level)
    torch.testing.assert_close(a.next_embedding_prediction[:,:7],b.next_embedding_prediction)

def test_same_episode_batch_is_stateless():
    m=model();x=torch.randn(1,10,8)
    a=m(x);m(torch.randn(1,5,8));b=m(x)
    torch.testing.assert_close(a.scaled_relative_level,b.scaled_relative_level,atol=0,rtol=0)

@pytest.mark.parametrize('x',[torch.zeros(8),torch.zeros(1,0,8),torch.zeros(1,3,7),torch.zeros(1,3,8,dtype=torch.long),torch.full((1,3,8),float('nan'))])
def test_invalid_inputs(x):
    with pytest.raises(ValueError):model()(x)

def test_save_reload():
    import io
    m=model();stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
    n=model();n.load_state_dict(torch.load(stream,weights_only=True))
    x=torch.randn(1,10,8)
    torch.testing.assert_close(m(x).scaled_relative_level,n(x).scaled_relative_level,atol=0,rtol=0)
