import numpy as np
import pytest
import torch

from state_interpreter import PredictiveGRUStateInterpreter
from state_interpreter.temporal_ranking import (
    full_window_distance_level,
    temporal_ranking_loss,
)


def test_ranking_has_correct_direction_and_penalizes_collapse():
    rising = torch.arange(20, dtype=torch.float32) * 0.1
    assert temporal_ranking_loss(rising, pair_lag=3, margin=0.2) == 0
    assert temporal_ranking_loss(-rising, pair_lag=3, margin=0.2) > 0.49
    flat = torch.zeros(20)
    assert float(temporal_ranking_loss(flat, pair_lag=3, margin=0.2)) == pytest.approx(0.2)


def test_full_window_matches_manual_distance_and_is_prefix_causal():
    torch.manual_seed(19)
    hidden = torch.randn(60, 8)
    actual = full_window_distance_level(hidden, calibration_steps=15, temporal_window=10)
    raw = torch.linalg.vector_norm(hidden - hidden[:15].mean(0), dim=1).numpy()
    expected = np.array([np.mean(raw[i-9:i+1]) for i in range(24, 60)])
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
    prefix = full_window_distance_level(hidden[:40], calibration_steps=15, temporal_window=10)
    torch.testing.assert_close(prefix, actual[:len(prefix)])


def test_ranking_backpropagates_through_gru():
    torch.manual_seed(23)
    model = PredictiveGRUStateInterpreter(embedding_dim=8)
    output = model(torch.randn(1, 60, 8))
    level = full_window_distance_level(output.hidden_sequence[0], calibration_steps=15, temporal_window=10)
    loss = temporal_ranking_loss(level, pair_lag=10, margin=0.05)
    loss.backward()
    gradients = [p.grad for p in model.gru.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_ranking_rejects_invalid_pairs_and_window():
    with pytest.raises(ValueError, match="pair_lag"):
        temporal_ranking_loss(torch.ones(10), pair_lag=10, margin=0.05)
    with pytest.raises(ValueError, match="margin"):
        temporal_ranking_loss(torch.ones(10), pair_lag=2, margin=-1)
    with pytest.raises(ValueError, match="short"):
        full_window_distance_level(torch.ones(20, 8), calibration_steps=15, temporal_window=10)
