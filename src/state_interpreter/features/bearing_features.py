"""XJTU-SY compatible 65-feature vibration representation.

The feature definitions follow the MIT-licensed ``thfmn/xjtu-sy-bearing``
engineering baseline at commit ``7d7231c582961741bde629da6731e6c169d88785``.
This is a small, dependency-focused reimplementation for the PyTorch-based
State Interpreter project; it does not include the upstream RUL labels,
onset annotations, training pipeline, or cloud tooling.

Upstream copyright (C) 2026 Tobias Hoffmann. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats
from scipy.fft import rfft, rfftfreq


SAMPLING_RATE = 25_600.0
CONDITION_SHAFT_HZ = {
    "35Hz12kN": 35.0,
    "37.5Hz11kN": 37.5,
    "40Hz10kN": 40.0,
}

TIME_FEATURES = (
    "mean",
    "std",
    "variance",
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "kurtosis",
    "skewness",
    "line_integral",
    "zero_crossing_rate",
    "entropy",
    "percentile_5",
    "percentile_50",
    "percentile_95",
)

FREQUENCY_FEATURES = (
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "band_power_0_1k",
    "band_power_1_3k",
    "band_power_3_6k",
    "band_power_6_12k",
    "dominant_frequency",
    "mean_frequency",
    "bpfo_band_power",
    "bpfi_band_power",
    "bsf_band_power",
    "ftf_band_power",
)

TIME_FEATURE_NAMES = tuple(f"h_{name}" for name in TIME_FEATURES) + tuple(
    f"v_{name}" for name in TIME_FEATURES
) + ("cross_correlation",)
FREQUENCY_FEATURE_NAMES = tuple(
    f"h_{name}" for name in FREQUENCY_FEATURES
) + tuple(f"v_{name}" for name in FREQUENCY_FEATURES)
FEATURE_NAMES = TIME_FEATURE_NAMES + FREQUENCY_FEATURE_NAMES
NUM_BEARING_FEATURES = len(FEATURE_NAMES)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else 0.0


def _entropy(values: np.ndarray) -> float:
    counts, _ = np.histogram(values, bins=100)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _time_features(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    mean_absolute = float(absolute.mean())
    rms = float(np.sqrt(np.mean(values**2)))
    peak = float(absolute.max())
    mean_sqrt_squared = float(np.mean(np.sqrt(absolute)) ** 2)
    std = float(np.std(values))
    if std < 1e-12:
        correlation_safe_kurtosis = 0.0
        skewness = 0.0
    else:
        correlation_safe_kurtosis = float(stats.kurtosis(values, fisher=True))
        skewness = float(stats.skew(values))
        if not np.isfinite(correlation_safe_kurtosis):
            correlation_safe_kurtosis = 0.0
        if not np.isfinite(skewness):
            skewness = 0.0
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return np.asarray(
        [
            float(np.mean(values)),
            std,
            float(np.var(values)),
            rms,
            peak,
            float(np.ptp(values)),
            _safe_ratio(peak, rms),
            _safe_ratio(rms, mean_absolute),
            _safe_ratio(peak, mean_absolute),
            _safe_ratio(peak, mean_sqrt_squared),
            correlation_safe_kurtosis,
            skewness,
            float(np.sum(np.abs(np.diff(values)))),
            float(np.count_nonzero(np.diff(np.sign(values))) / (values.size - 1)),
            _entropy(values),
            float(p5),
            float(p50),
            float(p95),
        ],
        dtype=np.float64,
    )


def calculate_bearing_frequencies(shaft_frequency_hz: float) -> dict[str, float]:
    """Return LDK UER204 characteristic frequencies for one shaft speed."""

    if shaft_frequency_hz <= 0:
        raise ValueError("shaft_frequency_hz must be positive")
    n_balls = 8
    ball_diameter = 7.92
    pitch_diameter = 34.5
    ratio = ball_diameter / pitch_diameter
    cosine = math.cos(0.0)
    return {
        "bpfo": (n_balls * shaft_frequency_hz / 2.0) * (1.0 - ratio * cosine),
        "bpfi": (n_balls * shaft_frequency_hz / 2.0) * (1.0 + ratio * cosine),
        "bsf": (pitch_diameter * shaft_frequency_hz / (2.0 * ball_diameter))
        * (1.0 - (ratio * cosine) ** 2),
        "ftf": (shaft_frequency_hz / 2.0) * (1.0 - ratio * cosine),
    }


def _band_power(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    return float(np.sum(magnitudes[mask] ** 2))


def _frequency_features(
    values: np.ndarray,
    sampling_rate: float,
    characteristic_frequencies: dict[str, float],
) -> np.ndarray:
    window = scipy_signal.get_window("hann", values.size)
    magnitudes = np.abs(rfft(values * window)) * 2.0 / values.size
    frequencies = rfftfreq(values.size, d=1.0 / sampling_rate)
    magnitude_sum = float(magnitudes.sum())
    if magnitude_sum <= 0:
        centroid = bandwidth = rolloff = dominant = 0.0
    else:
        centroid = float(np.sum(frequencies * magnitudes) / magnitude_sum)
        bandwidth = float(
            np.sqrt(np.sum(magnitudes * (frequencies - centroid) ** 2) / magnitude_sum)
        )
        cumulative = np.cumsum(magnitudes)
        rolloff_index = min(
            int(np.searchsorted(cumulative, 0.85 * magnitude_sum)),
            frequencies.size - 1,
        )
        rolloff = float(frequencies[rolloff_index])
        dominant = float(frequencies[int(np.argmax(magnitudes))])
    stabilized = magnitudes + 1e-10
    flatness = float(np.exp(np.mean(np.log(stabilized))) / np.mean(stabilized))
    psd_frequencies, psd = scipy_signal.welch(
        values,
        fs=sampling_rate,
        nperseg=min(1024, values.size),
    )
    psd_sum = float(psd.sum())
    mean_frequency = (
        float(np.sum(psd_frequencies * psd) / psd_sum) if psd_sum > 0 else 0.0
    )
    output = [
        centroid,
        bandwidth,
        rolloff,
        flatness,
        _band_power(frequencies, magnitudes, 0.0, 1_000.0),
        _band_power(frequencies, magnitudes, 1_000.0, 3_000.0),
        _band_power(frequencies, magnitudes, 3_000.0, 6_000.0),
        _band_power(frequencies, magnitudes, 6_000.0, 12_000.0),
        dominant,
        mean_frequency,
    ]
    for name in ("bpfo", "bpfi", "bsf", "ftf"):
        center = characteristic_frequencies[name]
        half_width = center * 0.1
        output.append(
            _band_power(
                frequencies,
                magnitudes,
                max(0.0, center - half_width),
                center + half_width,
            )
        )
    return np.asarray(output, dtype=np.float64)


@dataclass(frozen=True)
class XJTUBearingFeatureExtractor:
    """Extract the ordered 65-feature baseline from a dual-channel signal."""

    condition: str
    sampling_rate: float = SAMPLING_RATE

    def __post_init__(self) -> None:
        if self.condition not in CONDITION_SHAFT_HZ:
            raise ValueError(
                f"unknown condition {self.condition!r}; expected one of "
                f"{sorted(CONDITION_SHAFT_HZ)}"
            )
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def extract(self, vibration: np.ndarray) -> np.ndarray:
        """Return float32 features for input shape ``[samples, 2]``."""

        values = np.asarray(vibration, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                "vibration must have shape [samples, 2], "
                f"got {values.shape}"
            )
        if values.shape[0] < 2:
            raise ValueError("vibration must contain at least two samples")
        if not np.isfinite(values).all():
            raise ValueError("vibration contains NaN or infinite values")

        horizontal = values[:, 0]
        vertical = values[:, 1]
        h_std = float(horizontal.std())
        v_std = float(vertical.std())
        if h_std < 1e-12 or v_std < 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(horizontal, vertical)[0, 1])

        characteristic = calculate_bearing_frequencies(
            CONDITION_SHAFT_HZ[self.condition]
        )
        features = np.concatenate(
            [
                _time_features(horizontal),
                _time_features(vertical),
                np.asarray([correlation]),
                _frequency_features(horizontal, self.sampling_rate, characteristic),
                _frequency_features(vertical, self.sampling_rate, characteristic),
            ]
        ).astype(np.float32)
        if features.shape != (NUM_BEARING_FEATURES,):
            raise RuntimeError(f"expected 65 features, got {features.shape}")
        if not np.isfinite(features).all():
            raise RuntimeError("feature extraction produced NaN or infinite values")
        return features
