import pytest
import torch
from state_interpreter.forecast_evaluation import aligned_squared_errors, forecast_metrics


def test_perfect_forecast_and_unused_last_prediction():
    x = torch.tensor([[0.], [1.], [2.], [3.]])
    forecast = torch.tensor([[1.], [2.], [3.], [999.]])
    g, p = aligned_squared_errors(x, forecast)
    result = forecast_metrics(g, p, target_start_step=1)
    assert result["target_count"] == 3
    assert result["gru_mse"] == 0
    assert result["persistence_mse"] == 1
    assert result["skill"] == 1


def test_start_uses_target_not_origin_index():
    x = torch.tensor([[0.], [1.], [3.], [6.]])
    g, p = aligned_squared_errors(x, x)
    result = forecast_metrics(g, p, target_start_step=2)
    assert result["target_count"] == 2
    assert result["persistence_mse"] == 6.5  # differences 2 and 3
    assert result["skill"] == 0


def test_zero_baseline_is_explicitly_undefined_and_failure_is_negative():
    g, p = aligned_squared_errors(torch.ones(4, 2), torch.zeros(4, 2))
    assert forecast_metrics(g, p, target_start_step=1)["skill"] is None
    assert forecast_metrics(torch.full((3, 2), 4.), torch.ones(3, 2), target_start_step=1)["skill"] == -3


def test_invalid_input_and_target_rejected():
    with pytest.raises(ValueError, match="matching"):
        aligned_squared_errors(torch.ones(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="target_start"):
        forecast_metrics(torch.ones(3, 2), torch.ones(3, 2), target_start_step=4)
