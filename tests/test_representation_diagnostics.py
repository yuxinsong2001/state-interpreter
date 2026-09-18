import numpy as np

from state_interpreter.representation_diagnostics import (
    evenly_spaced_indices,
    feature_lifetime_spearman,
    nearest_centroid_predict,
    ridge_fit_predict,
)


def test_evenly_spaced_indices_include_endpoints() -> None:
    indices = evenly_spaced_indices(11, 6)
    assert indices.tolist() == [0, 2, 4, 6, 8, 10]


def test_feature_correlations_retain_direction() -> None:
    time = np.arange(20, dtype=float)
    features = np.column_stack((time, -time, np.ones_like(time)))
    result = feature_lifetime_spearman(features)
    assert np.allclose(result, [1.0, -1.0, 0.0])


def test_ridge_recovers_linear_mapping() -> None:
    x = np.arange(20, dtype=float)[:, None]
    prediction = ridge_fit_predict(x, 2 * x[:, 0] + 1, x, alpha=1e-6)
    assert np.max(np.abs(prediction - (2 * x[:, 0] + 1))) < 1e-5


def test_nearest_centroid_recovers_separated_identity() -> None:
    train_x = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = np.array([0, 0, 1, 1])
    prediction = nearest_centroid_predict(train_x, labels, np.array([[0.2], [9.9]]))
    assert prediction.tolist() == [0, 1]
