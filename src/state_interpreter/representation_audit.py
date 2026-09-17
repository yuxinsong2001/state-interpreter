"""Small deterministic helpers for latent representation audits."""

from __future__ import annotations

import numpy as np


def stage_labels(normalized_lifetime: np.ndarray, thresholds=(1 / 3, 2 / 3)) -> np.ndarray:
    values = np.asarray(normalized_lifetime, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional lifetime required")
    if not 0 < thresholds[0] < thresholds[1] < 1:
        raise ValueError("ordered thresholds inside (0,1) required")
    return np.digitize(values, thresholds).astype(int)


def early_center(
    embeddings: np.ndarray,
    bearing_ids: np.ndarray,
    step_ids: np.ndarray,
    calibration_steps: int,
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=float)
    bearing_ids = np.asarray(bearing_ids)
    step_ids = np.asarray(step_ids)
    if embeddings.ndim != 2 or len(embeddings) != len(bearing_ids) or len(step_ids) != len(bearing_ids):
        raise ValueError("aligned embeddings, bearing_ids and step_ids required")
    if calibration_steps <= 0 or not np.isfinite(embeddings).all():
        raise ValueError("positive calibration_steps and finite embeddings required")
    centered = np.empty_like(embeddings)
    for bearing in np.unique(bearing_ids):
        mask = bearing_ids == bearing
        reference = mask & (step_ids < calibration_steps)
        if int(reference.sum()) != calibration_steps:
            raise ValueError(f"bearing {bearing} lacks the complete calibration prefix")
        centered[mask] = embeddings[mask] - embeddings[reference].mean(axis=0)
    return centered


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("finite two-dimensional vectors required")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero direction vector")
    normalized = vectors / norms[:, None]
    return normalized @ normalized.T


def effective_dimension(explained_variance: np.ndarray) -> float:
    values = np.asarray(explained_variance, dtype=float)
    if values.ndim != 1 or np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("nonnegative finite variance with positive sum required")
    return float(values.sum() ** 2 / np.square(values).sum())
