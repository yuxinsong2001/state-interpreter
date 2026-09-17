import pytest
import torch

from state_interpreter.residual_gru import ResidualGRUStateInterpreter


def test_zero_initialised_model_is_exact_persistence():
    torch.manual_seed(7)
    model = ResidualGRUStateInterpreter(embedding_dim=3, hidden_dim=4)
    sequence = torch.randn(2, 6, 3)
    output = model(sequence)
    torch.testing.assert_close(output.next_embedding_prediction, sequence, atol=0, rtol=0)
    assert torch.count_nonzero(model.prediction_head.weight) == 0
    assert torch.count_nonzero(model.prediction_head.bias) == 0


def test_residual_output_adds_delta_to_current_input():
    model = ResidualGRUStateInterpreter(embedding_dim=2, hidden_dim=3)
    with torch.no_grad():
        model.prediction_head.bias.copy_(torch.tensor([0.25, -0.5]))
    sequence = torch.randn(1, 4, 2)
    output = model(sequence)
    expected = sequence + torch.tensor([0.25, -0.5])
    torch.testing.assert_close(output.next_embedding_prediction, expected)


def test_causal_prefix_predictions_are_identical():
    torch.manual_seed(9)
    model = ResidualGRUStateInterpreter(embedding_dim=2, hidden_dim=3)
    with torch.no_grad():
        model.prediction_head.weight.normal_()
    sequence = torch.randn(1, 8, 2)
    full = model(sequence).next_embedding_prediction[:, :5]
    prefix = model(sequence[:, :5]).next_embedding_prediction
    torch.testing.assert_close(full, prefix)


@pytest.mark.parametrize("shape", [(3, 2), (1, 0, 2), (1, 3, 4)])
def test_invalid_shapes_are_rejected(shape):
    model = ResidualGRUStateInterpreter(embedding_dim=2)
    with pytest.raises(ValueError):
        model(torch.zeros(shape))
