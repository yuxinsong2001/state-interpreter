import pytest
import torch
from state_interpreter.gru_pcgrad import project_two_gradients, pcgrad_shared_gradients
from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter


def test_known_projection():
    a, b = torch.tensor([1., 0.]), torch.tensor([-1., 1.])
    ap, bp, d = project_two_gradients(a, b)
    torch.testing.assert_close(ap, torch.tensor([.5, .5]))
    torch.testing.assert_close(bp, torch.tensor([0., 1.]))
    torch.testing.assert_close(ap+bp, torch.tensor([.5, 1.5]))
    assert d.projected and not d.zero_norm_skipped
    assert float(ap@b) == pytest.approx(0.)
    assert float(bp@a) == pytest.approx(0.)


def test_nonconflict_unchanged():
    a, b = torch.tensor([1., 2.]), torch.tensor([2., 1.])
    ap, bp, d = project_two_gradients(a, b)
    assert torch.equal(ap, a) and torch.equal(bp, b) and not d.projected


def test_zero_safe():
    ap, bp, d = project_two_gradients(torch.zeros(2), torch.ones(2))
    assert d.zero_norm_skipped and d.cosine is None
    assert torch.equal(ap+bp, torch.ones(2))


def test_antiparallel_zero_not_fallback():
    ap, bp, d = project_two_gradients(torch.tensor([1., 2.]), torch.tensor([-1., -2.]))
    assert d.projected and not d.zero_norm_skipped
    torch.testing.assert_close(ap+bp, torch.zeros(2))


@pytest.mark.parametrize('epsilon', [0, -1, float('nan')])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        project_two_gradients(torch.ones(2), torch.ones(2), epsilon=epsilon)


@pytest.mark.parametrize('a,b', [(torch.ones(2), torch.ones(3)),
                               (torch.tensor([float('nan')]), torch.ones(1)),
                               (torch.tensor([float('inf')]), torch.ones(1))])
def test_invalid_vectors(a,b):
    with pytest.raises(ValueError): project_two_gradients(a,b)


def model_losses():
    torch.manual_seed(42)
    model = GRUStateReadoutInterpreter(embedding_dim=8)
    with torch.no_grad(): model.core.prediction_head.weight.normal_(std=.1)
    z = torch.randn(1,12,8)
    out = model(z)
    return model, (out.next_embedding_prediction[:,:-1]-z[:,1:]).square().mean(), out.scaled_relative_level.square().mean()


def test_shared_formula_heads_and_no_update():
    m,p,l = model_losses()
    shared = list(m.core.gru.parameters())
    ph,lh = list(m.core.prediction_head.parameters()),list(m.level_head.parameters())
    a = torch.autograd.grad(p,shared+ph,retain_graph=True)
    b = torch.autograd.grad(l,shared+lh,retain_graph=True)
    n = len(shared)
    ap,bp,_ = project_two_gradients(torch.cat([x.flatten() for x in a[:n]]),torch.cat([x.flatten() for x in b[:n]]))
    before = {k:v.clone() for k,v in m.state_dict().items()}
    pcgrad_shared_gradients(m,p,l)
    torch.testing.assert_close(torch.cat([x.grad.flatten() for x in shared]),ap+bp)
    for param,g in zip(ph,a[n:]): torch.testing.assert_close(param.grad,g)
    for param,g in zip(lh,b[n:]): torch.testing.assert_close(param.grad,g)
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())


def test_adam_step():
    m,p,l = model_losses()
    before = m.core.gru.weight_ih_l0.detach().clone()
    optimizer = torch.optim.Adam(m.parameters(),lr=.001)
    pcgrad_shared_gradients(m,p,l)
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    optimizer.step()
    assert not torch.equal(before,m.core.gru.weight_ih_l0)


def test_nonfinite_no_partial_assignment():
    m,p,l = model_losses()
    with pytest.raises(ValueError): pcgrad_shared_gradients(m,p*float('nan'),l)
    assert all(param.grad is None for param in m.parameters())
