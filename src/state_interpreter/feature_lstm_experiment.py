"""Reproducible utilities for the preregistered Feature LSTM LOBO study."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

from state_interpreter.data.feature_sequences import build_bearing_feature_windows
from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler


@dataclass(frozen=True)
class PreparedBearingWindows:
    windows: torch.Tensor
    remaining_useful_life: torch.Tensor
    normalized_lifetime: torch.Tensor
    end_step_ids: torch.Tensor


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def prepare_bearing_windows(
    features: np.ndarray | torch.Tensor,
    *,
    bearing_id: str,
    calibration_steps: int,
    window_size: int,
    scaling_mode: str,
) -> PreparedBearingWindows:
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != 65:
        raise ValueError(f"expected [steps, 65], got {tuple(values.shape)}")
    if values.shape[0] < max(calibration_steps, window_size):
        raise ValueError("bearing sequence is too short")
    scaler = StableCausalFeatureScaler.fit(
        values,
        calibration_steps=calibration_steps,
        mode=scaling_mode,  # type: ignore[arg-type]
    )
    scaled = scaler.transform(values)
    steps = list(range(values.shape[0]))
    built = build_bearing_feature_windows(
        scaled,
        [bearing_id] * values.shape[0],
        steps,
        window_size=window_size,
    )
    endpoints = torch.tensor(built.end_step_ids, dtype=torch.long)
    lifetime = endpoints.to(torch.float32) / float(values.shape[0] - 1)
    rul = 1.0 - lifetime
    return PreparedBearingWindows(
        windows=built.windows,
        remaining_useful_life=rul,
        normalized_lifetime=lifetime,
        end_step_ids=endpoints,
    )


def backward_step_fraction(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("values must be a one-dimensional sequence of length >= 2")
    return float(np.mean(np.diff(array) < 0.0))
