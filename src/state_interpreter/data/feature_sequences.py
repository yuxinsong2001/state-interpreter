"""Causal calibration and bearing-safe windows for feature baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class CausalFeatureStandardizer:
    """Feature-wise statistics fitted from a fixed early calibration prefix."""

    mean: torch.Tensor
    std: torch.Tensor
    calibration_steps: int

    @classmethod
    def fit(
        cls,
        sequence: torch.Tensor,
        *,
        calibration_steps: int = 15,
        epsilon: float = 1e-6,
    ) -> "CausalFeatureStandardizer":
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape [steps, features]")
        if sequence.shape[1] == 0:
            raise ValueError("sequence must contain at least one feature")
        if calibration_steps < 2 or calibration_steps > sequence.shape[0]:
            raise ValueError(
                "calibration_steps must select at least two available steps"
            )
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not torch.isfinite(sequence).all():
            raise ValueError("sequence must contain only finite values")
        early = sequence[:calibration_steps]
        mean = early.mean(dim=0)
        std = early.std(dim=0, correction=0).clamp_min(epsilon)
        return cls(
            mean=mean.detach().clone(),
            std=std.detach().clone(),
            calibration_steps=calibration_steps,
        )

    def transform(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[1:] != self.mean.shape:
            raise ValueError(
                f"expected [steps, {self.mean.numel()}], got {tuple(sequence.shape)}"
            )
        if not torch.isfinite(sequence).all():
            raise ValueError("sequence must contain only finite values")
        return (sequence - self.mean.to(sequence)) / self.std.to(sequence)

    def inverse_transform(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[1:] != self.mean.shape:
            raise ValueError(
                f"expected [steps, {self.mean.numel()}], got {tuple(sequence.shape)}"
            )
        return sequence * self.std.to(sequence) + self.mean.to(sequence)


@dataclass(frozen=True)
class BearingFeatureWindows:
    """Feature windows with the bearing and endpoint retained for every row."""

    windows: torch.Tensor
    bearing_ids: tuple[str, ...]
    end_step_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.windows.ndim != 3:
            raise ValueError("windows must have shape [windows, steps, features]")
        if not (
            self.windows.shape[0]
            == len(self.bearing_ids)
            == len(self.end_step_ids)
        ):
            raise ValueError("window metadata length mismatch")


def build_bearing_feature_windows(
    features: torch.Tensor,
    bearing_ids: Sequence[str],
    step_ids: Sequence[int],
    *,
    window_size: int = 10,
) -> BearingFeatureWindows:
    """Sort within each bearing and create windows that never cross boundaries."""

    if features.ndim != 2:
        raise ValueError("features must have shape [measurements, features]")
    if features.shape[0] != len(bearing_ids) or features.shape[0] != len(step_ids):
        raise ValueError("features, bearing_ids, and step_ids must have equal length")
    if features.shape[0] == 0:
        raise ValueError("cannot build windows from empty features")
    if features.shape[1] == 0:
        raise ValueError("features must contain at least one column")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    if any(not bearing_id for bearing_id in bearing_ids):
        raise ValueError("bearing_ids must not be empty")
    if any(step_id < 0 for step_id in step_ids):
        raise ValueError("step_ids must be non-negative")

    grouped: dict[str, list[tuple[int, int]]] = {}
    bearing_order: list[str] = []
    for row_index, (bearing_id, step_id) in enumerate(zip(bearing_ids, step_ids)):
        if bearing_id not in grouped:
            grouped[bearing_id] = []
            bearing_order.append(bearing_id)
        grouped[bearing_id].append((int(step_id), row_index))

    windows: list[torch.Tensor] = []
    window_bearings: list[str] = []
    window_end_steps: list[int] = []
    for bearing_id in bearing_order:
        ordered = sorted(grouped[bearing_id], key=lambda item: item[0])
        ordered_steps = [item[0] for item in ordered]
        if len(ordered_steps) != len(set(ordered_steps)):
            raise ValueError(f"duplicate step_id in bearing {bearing_id}")
        if len(ordered) < window_size:
            continue
        row_indices = torch.tensor(
            [item[1] for item in ordered], dtype=torch.long, device=features.device
        )
        bearing_features = features.index_select(0, row_indices)
        for start in range(len(ordered) - window_size + 1):
            end = start + window_size
            windows.append(bearing_features[start:end])
            window_bearings.append(bearing_id)
            window_end_steps.append(ordered_steps[end - 1])

    if not windows:
        return BearingFeatureWindows(
            windows=features.new_empty((0, window_size, features.shape[1])),
            bearing_ids=(),
            end_step_ids=(),
        )
    return BearingFeatureWindows(
        windows=torch.stack(windows),
        bearing_ids=tuple(window_bearings),
        end_step_ids=tuple(window_end_steps),
    )
