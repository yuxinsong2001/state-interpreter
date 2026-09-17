import numpy as np
import pytest

from state_interpreter.representation_audit import (
    cosine_similarity_matrix,
    early_center,
    effective_dimension,
    stage_labels,
)


def test_stage_labels_use_three_fixed_bins():
    result = stage_labels(np.array([0.0, 0.2, 1 / 3, 0.5, 2 / 3, 1.0]))
    np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2])


def test_early_center_uses_each_bearing_prefix_only():
    embeddings = np.array([[1.0], [3.0], [5.0], [10.0], [14.0], [18.0]])
    bearings = np.array(["a", "a", "a", "b", "b", "b"])
    steps = np.array([0, 1, 2, 0, 1, 2])
    centered = early_center(embeddings, bearings, steps, calibration_steps=2)
    np.testing.assert_allclose(centered[:, 0], [-1, 1, 3, -2, 2, 6])


def test_cosine_matrix_preserves_direction_relationships():
    matrix = cosine_similarity_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]]))
    np.testing.assert_allclose(matrix, [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])


def test_effective_dimension_matches_equal_and_collapsed_variance():
    assert effective_dimension(np.ones(4)) == pytest.approx(4)
    assert effective_dimension(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)


def test_missing_calibration_prefix_is_rejected():
    with pytest.raises(ValueError):
        early_center(np.ones((2, 1)), np.array(["a", "a"]), np.array([0, 2]), 2)
