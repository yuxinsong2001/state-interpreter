import pytest
import torch

from state_interpreter.data import (
    CausalFeatureStandardizer,
    build_bearing_feature_windows,
)
from state_interpreter.encoders import FeatureLSTM


def test_causal_standardizer_uses_only_fixed_prefix() -> None:
    early = torch.tensor([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    first = torch.cat((early, torch.tensor([[10.0, 100.0]])))
    second = torch.cat((early, torch.tensor([[-999.0, -999.0]])))

    fitted_first = CausalFeatureStandardizer.fit(first, calibration_steps=3)
    fitted_second = CausalFeatureStandardizer.fit(second, calibration_steps=3)

    torch.testing.assert_close(fitted_first.mean, fitted_second.mean)
    torch.testing.assert_close(fitted_first.std, fitted_second.std)


def test_causal_standardizer_calibrates_prefix_and_handles_constant_features() -> None:
    sequence = torch.tensor(
        [[0.0, 7.0], [2.0, 7.0], [4.0, 7.0], [8.0, 9.0]],
        dtype=torch.float32,
    )
    standardizer = CausalFeatureStandardizer.fit(sequence, calibration_steps=3)

    transformed = standardizer.transform(sequence)

    torch.testing.assert_close(transformed[:3, 0].mean(), torch.tensor(0.0))
    torch.testing.assert_close(
        transformed[:3, 0].std(correction=0), torch.tensor(1.0)
    )
    assert torch.isfinite(transformed).all()
    torch.testing.assert_close(standardizer.inverse_transform(transformed), sequence)


def test_causal_standardizer_rejects_short_or_nonfinite_sequence() -> None:
    with pytest.raises(ValueError, match="calibration_steps"):
        CausalFeatureStandardizer.fit(torch.ones(3, 65), calibration_steps=4)
    values = torch.ones(3, 65)
    values[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        CausalFeatureStandardizer.fit(values, calibration_steps=2)


def test_windows_sort_steps_and_never_cross_bearing_boundaries() -> None:
    # Rows are intentionally interleaved and out of temporal order.
    features = torch.tensor([[12.0], [20.0], [10.0], [22.0], [11.0], [21.0]])
    bearing_ids = ["A", "B", "A", "B", "A", "B"]
    step_ids = [2, 0, 0, 2, 1, 1]

    result = build_bearing_feature_windows(
        features, bearing_ids, step_ids, window_size=2
    )

    assert result.bearing_ids == ("A", "A", "B", "B")
    assert result.end_step_ids == (1, 2, 1, 2)
    torch.testing.assert_close(
        result.windows.squeeze(-1),
        torch.tensor([[10.0, 11.0], [11.0, 12.0], [20.0, 21.0], [21.0, 22.0]]),
    )


def test_windows_return_stable_empty_shape_for_short_bearings() -> None:
    result = build_bearing_feature_windows(
        torch.ones(4, 65), ["A", "A", "B", "B"], [0, 1, 0, 1], window_size=3
    )

    assert result.windows.shape == (0, 3, 65)
    assert result.bearing_ids == ()


def test_windows_reject_duplicate_steps_within_bearing() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_bearing_feature_windows(
            torch.ones(2, 65), ["A", "A"], [0, 0], window_size=1
        )


def test_ten_step_windows_feed_feature_lstm() -> None:
    features = torch.randn(12, 65)
    windows = build_bearing_feature_windows(
        features, ["A"] * 12, list(range(12)), window_size=10
    )

    output = FeatureLSTM()(windows.windows)

    assert windows.windows.shape == (3, 10, 65)
    assert output.score.shape == (3, 1)
    assert output.hidden.shape == (3, 16)
