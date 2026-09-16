import pytest
import torch

from state_interpreter import PredictiveGRUStateInterpreter


def test_predictive_gru_returns_expected_shapes() -> None:
    model = PredictiveGRUStateInterpreter(embedding_dim=8, hidden_dim=6)
    sequence = torch.randn(4, 12, 8)

    output = model(sequence)

    assert output.hidden_sequence.shape == (4, 12, 6)
    assert output.next_embedding_prediction.shape == (4, 12, 8)
    assert output.final_hidden.shape == (1, 4, 6)


def test_predictive_gru_is_causal_for_prefix() -> None:
    torch.manual_seed(7)
    model = PredictiveGRUStateInterpreter(embedding_dim=3, hidden_dim=5).eval()
    sequence = torch.randn(1, 10, 3)

    full = model(sequence)
    prefix = model(sequence[:, :6])

    torch.testing.assert_close(
        prefix.hidden_sequence,
        full.hidden_sequence[:, :6],
    )
    torch.testing.assert_close(
        prefix.next_embedding_prediction,
        full.next_embedding_prediction[:, :6],
    )


def test_predictive_gru_supports_next_step_training() -> None:
    model = PredictiveGRUStateInterpreter(embedding_dim=2, hidden_dim=4)
    sequence = torch.randn(2, 9, 2)
    output = model(sequence)
    loss = torch.nn.functional.mse_loss(
        output.next_embedding_prediction[:, :-1], sequence[:, 1:]
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_predictive_gru_rejects_invalid_inputs() -> None:
    model = PredictiveGRUStateInterpreter(embedding_dim=3)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="embedding_dim"):
        model(torch.zeros(1, 4, 2))
    with pytest.raises(ValueError, match="floating"):
        model(torch.zeros(1, 4, 3, dtype=torch.long))
    invalid = torch.zeros(1, 4, 3)
    invalid[0, 2, 1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(invalid)
