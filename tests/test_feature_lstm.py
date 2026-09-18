import pytest
import torch

from state_interpreter.encoders import FeatureLSTM


def test_feature_lstm_shapes_and_parameter_count() -> None:
    model = FeatureLSTM()
    windows = torch.randn(4, 10, 65)

    result = model(windows)

    assert result.score.shape == (4, 1)
    assert result.hidden.shape == (4, 16)
    # PyTorch stores input and recurrent LSTM biases separately. The equivalent
    # Keras model reports 11,041 parameters; this port therefore has 128 more.
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_169


def test_feature_lstm_supports_backward_pass() -> None:
    model = FeatureLSTM()
    result = model(torch.randn(3, 10, 65))

    result.score.square().mean().backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_feature_lstm_rejects_wrong_shape() -> None:
    model = FeatureLSTM()
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 10, 64))
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(2, 65))
