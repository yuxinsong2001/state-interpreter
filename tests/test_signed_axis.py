import numpy as np
import pytest

from state_interpreter.signed_axis import causal_mean, fit_signed_axis, signed_level


def test_axis_is_unit_and_oriented_toward_training_end() -> None:
    time = np.arange(20, dtype=float)
    sequences = [np.column_stack((time + shift, 0.1 * time)) for shift in (0, 3, 7)]
    axis = fit_signed_axis(sequences, calibration_steps=4)
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert axis[0] > 0


def test_signed_level_preserves_backward_direction() -> None:
    sequence = np.array([[0.0], [0.0], [1.0], [-1.0]])
    level = signed_level(sequence, np.array([1.0]), calibration_steps=2, temporal_window=1)
    assert np.isnan(level[:2]).all()
    assert level[2] == pytest.approx(1.0)
    assert level[3] == pytest.approx(-1.0)


def test_signed_level_is_prefix_causal() -> None:
    sequence = np.arange(12, dtype=float)[:, None]
    full = signed_level(sequence, np.ones(1), calibration_steps=3, temporal_window=4)
    prefix = signed_level(sequence[:8], np.ones(1), calibration_steps=3, temporal_window=4)
    np.testing.assert_allclose(full[:8], prefix, equal_nan=True)


def test_causal_mean_expands_then_uses_fixed_window() -> None:
    np.testing.assert_allclose(causal_mean(np.array([1.0, 3.0, 8.0]), 2), [1, 2, 5.5])


def test_invalid_training_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_signed_axis([], calibration_steps=2)

