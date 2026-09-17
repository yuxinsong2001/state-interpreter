"""Integrated unlabeled relative state: signed Level, Trend and GRU surprise."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple, Sequence

import numpy as np
import torch
from torch import nn


class RelativeStateV2Output(NamedTuple):
    level: float
    trend: float
    movement: float
    state: np.ndarray


def causal_slope(values: Sequence[float]) -> float:
    """Least-squares slope over an ordered causal window."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all() or len(y) == 0:
        raise ValueError("values must be a non-empty finite vector")
    if len(y) == 1:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float(np.sum(x * (y - y.mean())) / np.sum(x**2))


class RelativeStateInterpreterV2:
    """Online relative state using frozen axis and residual-GRU ensemble.

    A new episode is calibrated by its first embeddings. Later outputs use only
    current/past inputs. Movement is prediction surprise in standardized latent
    space relative to a fixed training-only reference value.
    """

    state_dim = 3

    def __init__(
        self,
        *,
        axis: np.ndarray,
        residual_models: Sequence[nn.Module],
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        movement_reference: float,
        calibration_steps: int = 15,
        temporal_window: int = 10,
    ) -> None:
        direction = np.asarray(axis, dtype=float)
        if direction.ndim != 1 or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0:
            raise ValueError("axis must be a finite non-zero vector")
        if not residual_models:
            raise ValueError("at least one residual model is required")
        if feature_mean.shape != feature_std.shape or tuple(feature_mean.shape) != direction.shape:
            raise ValueError("normalization and axis dimensions must match")
        if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all() or not torch.all(feature_std > 0):
            raise ValueError("normalization tensors must be finite and std positive")
        if movement_reference <= 0 or not np.isfinite(movement_reference):
            raise ValueError("movement_reference must be finite and positive")
        if calibration_steps < 2 or temporal_window < 2:
            raise ValueError("calibration_steps and temporal_window must be at least two")
        self.axis = direction / np.linalg.norm(direction)
        self.models = list(residual_models)
        for model in self.models:
            model.eval()
        self.feature_mean = feature_mean.detach().float().clone()
        self.feature_std = feature_std.detach().float().clone()
        self.movement_reference = float(movement_reference)
        self.calibration_steps = calibration_steps
        self.temporal_window = temporal_window
        self.reset()

    def reset(self) -> None:
        self._raw: list[torch.Tensor] = []
        self._baseline: torch.Tensor | None = None
        self._raw_level_history: deque[float] = deque(maxlen=self.temporal_window)
        self._level_history: deque[float] = deque(maxlen=self.temporal_window)

    @property
    def ready(self) -> bool:
        return self._baseline is not None

    def update(self, z: torch.Tensor) -> RelativeStateV2Output | None:
        if z.ndim != 1 or z.numel() != len(self.axis) or not torch.is_floating_point(z):
            raise ValueError("z must be a floating vector matching the axis")
        if not torch.isfinite(z).all():
            raise ValueError("z must be finite")
        current = z.detach().float().clone()
        self._raw.append(current)
        if self._baseline is None:
            if len(self._raw) == self.calibration_steps:
                self._baseline = torch.stack(self._raw).mean(0)
            return None

        centered = torch.stack(self._raw) - self._baseline
        standardized = (centered - self.feature_mean) / self.feature_std
        with torch.no_grad():
            forecasts = [model(standardized[:-1][None]).next_embedding_prediction[0, -1] for model in self.models]
        forecast = torch.stack(forecasts).mean(0)
        surprise = torch.linalg.vector_norm(standardized[-1] - forecast).item()
        raw_level = float((current - self._baseline).double().numpy() @ self.axis)
        self._raw_level_history.append(raw_level)
        level = float(np.mean(self._raw_level_history))
        self._level_history.append(level)
        trend = causal_slope(tuple(self._level_history))
        movement = surprise / self.movement_reference
        state = np.asarray([level, trend, movement], dtype=float)
        return RelativeStateV2Output(level, trend, movement, state)

