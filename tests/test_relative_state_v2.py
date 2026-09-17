import numpy as np
import pytest
import torch

from state_interpreter.relative_state_v2 import RelativeStateInterpreterV2, causal_slope
from state_interpreter.residual_gru import ResidualGRUStateInterpreter


def persistence_model(dim: int = 2) -> ResidualGRUStateInterpreter:
    return ResidualGRUStateInterpreter(embedding_dim=dim, hidden_dim=2)


def make_interpreter() -> RelativeStateInterpreterV2:
    return RelativeStateInterpreterV2(
        axis=np.array([1.0, 0.0]), residual_models=[persistence_model()],
        feature_mean=torch.zeros(2), feature_std=torch.ones(2), movement_reference=1.0,
        calibration_steps=2, temporal_window=3,
    )


def test_calibration_then_signed_level_trend_and_surprise() -> None:
    interpreter = make_interpreter()
    assert interpreter.update(torch.tensor([0.0, 0.0])) is None
    assert interpreter.update(torch.tensor([0.0, 0.0])) is None
    first = interpreter.update(torch.tensor([1.0, 0.0]))
    second = interpreter.update(torch.tensor([3.0, 0.0]))
    assert first is not None and second is not None
    assert first.level == pytest.approx(1.0)
    assert first.trend == pytest.approx(0.0)
    assert first.movement == pytest.approx(1.0)
    assert second.level == pytest.approx(2.0)
    assert second.trend == pytest.approx(1.0)
    assert second.movement == pytest.approx(2.0)


def test_signed_level_can_move_backward() -> None:
    interpreter = make_interpreter()
    for value in ([0.0, 0.0], [0.0, 0.0]): interpreter.update(torch.tensor(value))
    assert interpreter.update(torch.tensor([-2.0, 0.0])).level == pytest.approx(-2.0)


def test_prefix_causality() -> None:
    values = [torch.tensor([float(i), 0.0]) for i in range(7)]
    full = make_interpreter(); prefix = make_interpreter()
    full_outputs = [full.update(v) for v in values]
    prefix_outputs = [prefix.update(v) for v in values[:5]]
    for left, right in zip(full_outputs[:5], prefix_outputs):
        if left is None: assert right is None
        else: np.testing.assert_allclose(left.state, right.state)


def test_reset_removes_episode_state() -> None:
    interpreter = make_interpreter()
    for value in ([0.0, 0.0], [0.0, 0.0], [2.0, 0.0]): interpreter.update(torch.tensor(value))
    interpreter.reset()
    assert not interpreter.ready
    assert interpreter.update(torch.tensor([10.0, 0.0])) is None


def test_causal_slope() -> None:
    assert causal_slope([2.0]) == 0.0
    assert causal_slope([1.0, 3.0, 5.0]) == pytest.approx(2.0)


def test_rejects_invalid_movement_reference() -> None:
    with pytest.raises(ValueError):
        RelativeStateInterpreterV2(axis=np.ones(2), residual_models=[persistence_model()], feature_mean=torch.zeros(2), feature_std=torch.ones(2), movement_reference=0.0)

