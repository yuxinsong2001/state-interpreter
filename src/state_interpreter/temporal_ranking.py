"""Weak temporal supervision for a predictive GRU's existing distance readout."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def full_window_distance_level(
    hidden: torch.Tensor, *, calibration_steps: int, temporal_window: int
) -> torch.Tensor:
    """Differentiable causal Level, starting at c + w - 1 (zero-based).

    The early reference is fixed after calibration. Every returned average uses
    only the current and preceding post-calibration distances. No future values,
    end-of-life information, or time index are model inputs.
    """
    if hidden.ndim != 2 or not torch.is_floating_point(hidden):
        raise ValueError("hidden must be a floating [time, features] tensor")
    if calibration_steps < 1 or temporal_window < 1:
        raise ValueError("calibration_steps and temporal_window must be positive")
    if len(hidden) < calibration_steps + temporal_window:
        raise ValueError("sequence too short for a full post-calibration window")
    reference = hidden[:calibration_steps].mean(dim=0)
    distances = torch.linalg.vector_norm(hidden[calibration_steps:] - reference, dim=1)
    return F.avg_pool1d(distances[None, None], temporal_window, stride=1)[0, 0]


def temporal_ranking_loss(
    level: torch.Tensor, *, pair_lag: int, margin: float
) -> torch.Tensor:
    """Mean hinge penalty for later-minus-earlier Level below a fixed margin.

    This encodes a weak chronological progression assumption. It is not a
    physical damage label and is not valid evidence of health semantics by itself.
    """
    if level.ndim != 1 or not torch.is_floating_point(level):
        raise ValueError("level must be a floating vector")
    if pair_lag < 1 or len(level) <= pair_lag:
        raise ValueError("pair_lag must be positive and smaller than sequence length")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    return torch.relu(margin - (level[pair_lag:] - level[:-pair_lag])).mean()
