"""Training-only signed degradation-axis readout for latent trajectories."""

from __future__ import annotations

import numpy as np


def _validate_sequence(sequence: np.ndarray, calibration_steps: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("sequence must have shape [time, features]")
    if not np.isfinite(values).all():
        raise ValueError("sequence must contain only finite values")
    if calibration_steps < 1 or calibration_steps >= len(values):
        raise ValueError("calibration_steps must be inside the sequence")
    return values


def fit_signed_axis(
    training_sequences: list[np.ndarray], *, calibration_steps: int
) -> np.ndarray:
    """Fit PC1 on early-centred training trajectories and orient it forward.

    Orientation uses only the mean early-to-late displacement of the supplied
    training bearings. No held-out trajectory is needed.
    """
    if not training_sequences:
        raise ValueError("at least one training sequence is required")
    centered: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    feature_count: int | None = None
    for sequence in training_sequences:
        values = _validate_sequence(sequence, calibration_steps)
        if feature_count is None:
            feature_count = values.shape[1]
        if values.shape[1] != feature_count:
            raise ValueError("all sequences must have the same feature count")
        reference = values[:calibration_steps].mean(axis=0)
        current = values - reference
        centered.append(current)
        late_count = max(calibration_steps, int(np.ceil(0.2 * len(values))))
        displacements.append(current[-late_count:].mean(axis=0))

    matrix = np.concatenate(centered, axis=0)
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    axis = right_vectors[0].copy()
    forward = np.mean(displacements, axis=0)
    if float(axis @ forward) < 0:
        axis *= -1.0
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("could not estimate a non-zero degradation axis")
    return axis / norm


def causal_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return an expanding-then-fixed causal mean without future samples."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or window < 1:
        raise ValueError("values must be one-dimensional and window positive")
    output = np.empty_like(array)
    for index in range(len(array)):
        output[index] = np.mean(array[max(0, index - window + 1) : index + 1])
    return output


def signed_level(
    sequence: np.ndarray,
    axis: np.ndarray,
    *,
    calibration_steps: int,
    temporal_window: int,
) -> np.ndarray:
    """Project a new bearing relative to its own early reference, causally."""
    values = _validate_sequence(sequence, calibration_steps)
    direction = np.asarray(axis, dtype=float)
    if direction.shape != (values.shape[1],) or not np.isfinite(direction).all():
        raise ValueError("axis has an invalid shape or value")
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("axis must be non-zero")
    reference = values[:calibration_steps].mean(axis=0)
    raw = (values - reference) @ (direction / norm)
    output = np.full(len(values), np.nan, dtype=float)
    output[calibration_steps:] = causal_mean(
        raw[calibration_steps:], temporal_window
    )
    return output

