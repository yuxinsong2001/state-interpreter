import numpy as np
import pytest
import torch

from state_interpreter.feature_lstm_experiment import (
    backward_step_fraction,
    prepare_bearing_windows,
    set_deterministic_seed,
)


def test_prepared_windows_have_endpoint_rul_targets() -> None:
    features = np.arange(20 * 65, dtype=np.float32).reshape(20, 65)
    prepared = prepare_bearing_windows(
        features,
        bearing_id="Bearing3_1",
        calibration_steps=5,
        window_size=4,
        scaling_mode="signed_log1p",
    )
    assert prepared.windows.shape == (17, 4, 65)
    assert prepared.end_step_ids.tolist() == list(range(3, 20))
    assert torch.allclose(
        prepared.remaining_useful_life + prepared.normalized_lifetime,
        torch.ones(17),
    )
    assert float(prepared.remaining_useful_life[-1]) == pytest.approx(0.0)


def test_preparation_is_bearing_local_and_finite() -> None:
    features = np.ones((15, 65), dtype=np.float32)
    features[-1, 0] = 1e9
    prepared = prepare_bearing_windows(
        features,
        bearing_id="Bearing3_1",
        calibration_steps=5,
        window_size=10,
        scaling_mode="signed_log1p",
    )
    assert torch.isfinite(prepared.windows).all()
    assert float(prepared.windows.abs().max()) < 40.0


def test_backward_fraction() -> None:
    assert backward_step_fraction(np.array([0.0, 1.0, 0.5, 2.0])) == pytest.approx(
        1 / 3
    )


def test_seed_repeats_torch_values() -> None:
    set_deterministic_seed(17)
    first = torch.rand(4)
    set_deterministic_seed(17)
    second = torch.rand(4)
    assert torch.equal(first, second)
