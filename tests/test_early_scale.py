import numpy as np
import pytest

from state_interpreter.early_scale import apply_scale, early_scale


def test_none_scale_is_one() -> None:
    sequence = np.arange(12, dtype=float).reshape(6, 2)
    assert early_scale(sequence, np.array([1.0, 0.0]), calibration_steps=3, method="none") == 1.0


def test_axis_std_uses_only_axis_projection() -> None:
    sequence = np.array([[0.0, 100.0], [1.0, -100.0], [2.0, 50.0], [999.0, 999.0]])
    scale = early_scale(sequence, np.array([1.0, 0.0]), calibration_steps=3, method="early_axis_std")
    assert scale == pytest.approx(1.0)


def test_latent_rms_uses_early_centered_norm() -> None:
    sequence = np.array([[-1.0, 0.0], [1.0, 0.0], [100.0, 100.0]])
    scale = early_scale(sequence, np.array([1.0, 0.0]), calibration_steps=2, method="early_latent_rms")
    assert scale == pytest.approx(1.0)


def test_late_samples_do_not_change_scale() -> None:
    prefix = np.array([[0.0], [1.0], [2.0]])
    first = early_scale(prefix, np.ones(1), calibration_steps=2, method="early_axis_std")
    second = early_scale(np.vstack([prefix, [[1000.0]]]), np.ones(1), calibration_steps=2, method="early_axis_std")
    assert first == pytest.approx(second)


def test_apply_scale_uses_training_floor() -> None:
    values = np.array([np.nan, 2.0, 4.0])
    np.testing.assert_allclose(apply_scale(values, 0.01, floor=0.5), [np.nan, 4.0, 8.0], equal_nan=True)

