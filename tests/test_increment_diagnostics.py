import pytest
import torch

from state_interpreter.increment_diagnostics import aligned_increments, increment_metrics


def test_alignment_uses_same_origin_and_requested_target_start():
    sequence = torch.tensor([[0.0], [1.0], [3.0], [6.0]])
    forecasts = torch.tensor([[1.5], [2.5], [7.0], [99.0]])
    observed, predicted = aligned_increments(sequence, forecasts, target_start_step=2)
    torch.testing.assert_close(observed, torch.tensor([[2.0], [3.0]]))
    torch.testing.assert_close(predicted, torch.tensor([[1.5], [4.0]]))


def test_perfect_increment_prediction():
    observed = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
    metrics = increment_metrics(observed, observed.clone())
    assert metrics["raw_mse"] == 0
    assert metrics["oracle_bias_corrected_mse"] == 0
    assert metrics["flattened_cosine"] == pytest.approx(1)
    assert metrics["magnitude_ratio"] == pytest.approx(1)
    assert metrics["positive_dot_fraction"] == 1


def test_zero_increment_prediction_equals_persistence():
    observed = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
    metrics = increment_metrics(observed, torch.zeros_like(observed))
    assert metrics["raw_mse_ratio"] == pytest.approx(1)
    assert metrics["raw_skill"] == pytest.approx(0)
    assert metrics["magnitude_ratio"] == pytest.approx(0)
    assert metrics["flattened_cosine"] is None
    assert metrics["positive_dot_fraction"] == 0


def test_constant_error_is_removed_only_by_oracle_bias_correction():
    observed = torch.tensor([[1.0, -2.0], [2.0, 1.0], [-1.0, 3.0]])
    predicted = observed + torch.tensor([2.0, -4.0])
    metrics = increment_metrics(observed, predicted)
    assert metrics["bias_fraction_of_raw_mse"] == pytest.approx(1)
    assert metrics["oracle_bias_corrected_mse"] == pytest.approx(0)


def test_opposite_direction_is_detected():
    observed = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
    metrics = increment_metrics(observed, -observed)
    assert metrics["flattened_cosine"] == pytest.approx(-1)
    assert metrics["through_origin_gain"] == pytest.approx(-1)
    assert metrics["positive_dot_fraction"] == 0


@pytest.mark.parametrize("start", [0, 4])
def test_invalid_target_start_is_rejected(start):
    sequence = torch.zeros(4, 2)
    with pytest.raises(ValueError):
        aligned_increments(sequence, sequence.clone(), target_start_step=start)
