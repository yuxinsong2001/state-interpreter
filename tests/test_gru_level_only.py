import torch
from state_interpreter.gru_state_readout import GRUStateReadoutInterpreter


def test_level_only_updates_shared_and_readout_not_prediction():
    torch.manual_seed(20260916)
    model = GRUStateReadoutInterpreter(embedding_dim=8,hidden_dim=8)
    initial_head = {k:v.clone() for k,v in model.core.prediction_head.state_dict().items()}
    initial_shared = model.core.gru.weight_ih_l0.detach().clone()
    for parameter in model.core.prediction_head.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-5)
    z = torch.randn(1,40,8)
    output = model(z)
    torch.testing.assert_close(output.next_embedding_prediction,z,rtol=0,atol=0)
    output.scaled_relative_level[:,25:].square().mean().backward()
    assert all(p.grad is None for p in model.core.prediction_head.parameters())
    assert all(p.grad is not None for p in model.core.gru.parameters())
    assert all(p.grad is not None for p in model.level_head.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    assert not torch.equal(initial_shared,model.core.gru.weight_ih_l0)
    assert all(torch.equal(initial_head[k],v) for k,v in model.core.prediction_head.state_dict().items())
    torch.testing.assert_close(model(z).next_embedding_prediction,z,rtol=0,atol=0)
