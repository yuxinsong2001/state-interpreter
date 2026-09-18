import pytest
import torch

from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler


def _sequence() -> torch.Tensor:
    early = torch.tensor([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    late = torch.tensor([[1_000_001.0, 8.0]])
    return torch.cat((early, late), dim=0)


def test_signed_log_stabilizes_near_constant_feature_and_preserves_sign() -> None:
    scaler = StableCausalFeatureScaler.fit(
        _sequence(), calibration_steps=3, mode="signed_log1p"
    )
    output = scaler.transform(_sequence())
    assert torch.isfinite(output).all()
    assert output[-1, 0] > 0
    assert output[-1, 0] < 30


def test_signed_log_is_strictly_monotonic() -> None:
    sequence = torch.arange(1.0, 7.0).unsqueeze(1)
    scaler = StableCausalFeatureScaler.fit(
        sequence, calibration_steps=3, mode="signed_log1p"
    )
    output = scaler.transform(sequence).squeeze(1)
    assert torch.all(output[1:] > output[:-1])


def test_clip_has_fixed_bound() -> None:
    scaler = StableCausalFeatureScaler.fit(
        _sequence(), calibration_steps=3, mode="clip_20"
    )
    output = scaler.transform(_sequence())
    assert float(output.abs().max()) == pytest.approx(20.0)


def test_baseline_remains_available_as_control() -> None:
    scaler = StableCausalFeatureScaler.fit(
        _sequence(), calibration_steps=3, mode="baseline_zscore"
    )
    output = scaler.transform(_sequence())
    assert output[-1, 0] > 1e6


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        StableCausalFeatureScaler.fit(
            _sequence(), calibration_steps=3, mode="unknown"  # type: ignore[arg-type]
        )
