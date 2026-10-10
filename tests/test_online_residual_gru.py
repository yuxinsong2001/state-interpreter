import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'state-interpreter/src'))
import pytest
import torch
from state_interpreter.residual_gru import ResidualGRUStateInterpreter
try:
    from state_interpreter.online_residual_gru import OnlineResidualGRU
except ModuleNotFoundError:
    from online_residual_gru import OnlineResidualGRU

def test_stream_matches_batch_and_target_alignment():
    torch.manual_seed(42)
    m=ResidualGRUStateInterpreter(embedding_dim=8)
    with torch.no_grad():m.prediction_head.weight.normal_(std=.1)
    seq=torch.randn(1,20,8);expected=m(seq).next_embedding_prediction.detach()[0]
    online=OnlineResidualGRU(m)
    for i,z in enumerate(seq[0]):
        out=online.update(z)
        torch.testing.assert_close(out.next_prediction,expected[i],atol=1e-6,rtol=1e-5)
        assert out.target_step==i
        if i==0:assert out.prediction_error_mse is None and out.persistence_error_mse is None
        else:
            assert out.prediction_error_mse==pytest.approx(float((z-expected[i-1]).square().mean()),abs=1e-6)
            assert out.persistence_error_mse==pytest.approx(float((z-seq[0,i-1]).square().mean()))
    assert m.training

def test_reset_and_exact_initial_persistence():
    m=ResidualGRUStateInterpreter(embedding_dim=8);o=OnlineResidualGRU(m)
    for z in torch.randn(10,8):
        r=o.update(z);torch.testing.assert_close(r.next_prediction,z,atol=0,rtol=0)
        assert r.prediction_error_mse==r.persistence_error_mse
    o.reset();r=o.update(torch.ones(8));assert r.target_step==0 and r.prediction_error_mse is None

@pytest.mark.parametrize('z',[torch.zeros(2,8),torch.zeros(7),torch.full((8,),float('nan')),torch.zeros(8,dtype=torch.float64)])
def test_invalid(z):
    with pytest.raises(ValueError):OnlineResidualGRU(ResidualGRUStateInterpreter(embedding_dim=8)).update(z)
