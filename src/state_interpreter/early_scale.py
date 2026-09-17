"""Causal per-bearing scale estimates from an early calibration window."""

from __future__ import annotations

import numpy as np


METHODS = ("none", "early_axis_std", "early_latent_rms")


def early_scale(
    sequence: np.ndarray,
    axis: np.ndarray,
    *,
    calibration_steps: int,
    method: str,
) -> float:
    """Estimate a positive scale using only the early calibration samples."""
    values = np.asarray(sequence, dtype=float)
    direction = np.asarray(axis, dtype=float)
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if values.ndim != 2 or direction.shape != (values.shape[1],):
        raise ValueError("invalid sequence or axis shape")
    if calibration_steps < 2 or calibration_steps > len(values):
        raise ValueError("calibration_steps must contain at least two samples")
    if not np.isfinite(values).all() or not np.isfinite(direction).all():
        raise ValueError("inputs must be finite")
    if method == "none":
        return 1.0
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0:
        raise ValueError("axis must be non-zero")
    early = values[:calibration_steps]
    centered = early - early.mean(axis=0)
    if method == "early_axis_std":
        return float(np.std(centered @ (direction / direction_norm), ddof=1))
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def apply_scale(level: np.ndarray, scale: float, *, floor: float) -> np.ndarray:
    """Scale a Level trajectory using a training-only positive floor."""
    values = np.asarray(level, dtype=float)
    if scale < 0 or floor <= 0 or not np.isfinite([scale, floor]).all():
        raise ValueError("scale and floor must be finite and non-negative/positive")
    return values / max(scale, floor)

