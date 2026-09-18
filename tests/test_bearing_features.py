import numpy as np
import pytest

from state_interpreter.features import (
    FEATURE_NAMES,
    NUM_BEARING_FEATURES,
    XJTUBearingFeatureExtractor,
    calculate_bearing_frequencies,
)


def test_feature_extractor_returns_ordered_65_finite_features() -> None:
    rng = np.random.default_rng(20260918)
    vibration = rng.normal(size=(4096, 2))
    extractor = XJTUBearingFeatureExtractor("35Hz12kN")

    features = extractor.extract(vibration)

    assert features.shape == (65,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert NUM_BEARING_FEATURES == len(FEATURE_NAMES) == 65
    assert FEATURE_NAMES[0] == "h_mean"
    assert FEATURE_NAMES[36] == "cross_correlation"
    assert FEATURE_NAMES[-1] == "v_ftf_band_power"


def test_constant_signal_produces_no_nan_or_inf() -> None:
    vibration = np.ones((2048, 2), dtype=np.float64)

    features = XJTUBearingFeatureExtractor("37.5Hz11kN").extract(vibration)

    assert np.isfinite(features).all()
    assert features[36] == 0.0


def test_channels_remain_separate() -> None:
    time = np.arange(4096) / 25_600.0
    vibration = np.column_stack(
        (np.sin(2 * np.pi * 100 * time), np.sin(2 * np.pi * 800 * time))
    )

    features = XJTUBearingFeatureExtractor("40Hz10kN").extract(vibration)

    h_dominant = features[37 + 8]
    v_dominant = features[37 + 14 + 8]
    assert h_dominant == pytest.approx(100.0, abs=7.0)
    assert v_dominant == pytest.approx(800.0, abs=7.0)


@pytest.mark.parametrize("shaft_hz", [35.0, 37.5, 40.0])
def test_characteristic_frequencies_are_positive_and_ordered(shaft_hz: float) -> None:
    frequencies = calculate_bearing_frequencies(shaft_hz)

    assert set(frequencies) == {"bpfo", "bpfi", "bsf", "ftf"}
    assert all(value > 0 for value in frequencies.values())
    assert frequencies["bpfi"] > frequencies["bpfo"] > frequencies["ftf"]


def test_feature_extractor_rejects_invalid_input() -> None:
    extractor = XJTUBearingFeatureExtractor("35Hz12kN")
    with pytest.raises(ValueError, match="shape"):
        extractor.extract(np.ones((128, 1)))
    with pytest.raises(ValueError, match="NaN"):
        extractor.extract(np.asarray([[0.0, 1.0], [np.nan, 2.0]]))
