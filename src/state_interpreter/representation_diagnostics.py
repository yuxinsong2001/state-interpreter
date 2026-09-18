"""Small deterministic probes for cross-bearing representation diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    if length < 2 or count < 2 or count > length:
        raise ValueError("require 2 <= count <= length")
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def feature_lifetime_spearman(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("features must have shape [steps, dimensions]")
    lifetime = np.linspace(0.0, 1.0, values.shape[0])
    correlations = np.empty(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        if values[:, index].std() < 1e-12:
            correlations[index] = 0.0
            continue
        statistic = spearmanr(lifetime, values[:, index]).statistic
        correlations[index] = 0.0 if not np.isfinite(statistic) else statistic
    return correlations


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    z = np.asarray(test_x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (x - mean) / std
    test_standardized = (z - mean) / std
    design = np.column_stack((np.ones(x.shape[0]), standardized))
    test_design = np.column_stack((np.ones(z.shape[0]), test_standardized))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return test_design @ coefficients


def nearest_centroid_predict(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    labels = np.unique(train_labels)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    train = (train_x - mean) / std
    test = (test_x - mean) / std
    centroids = np.stack([train[train_labels == label].mean(axis=0) for label in labels])
    distances = ((test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(distances, axis=1)]
