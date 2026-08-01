import pytest
import torch

from state_interpreter import (
    InterpreterPhase,
    RelativeTemporalStateInterpreter,
)


def test_calibration_lifecycle_and_first_ready_state() -> None:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=1, calibration_steps=2, temporal_window=3
    )

    assert interpreter.phase is InterpreterPhase.CALIBRATING
    assert interpreter.update(torch.tensor([0.0])) is None
    assert interpreter.update(torch.tensor([2.0])) is None
    assert interpreter.phase is InterpreterPhase.READY
    assert interpreter.ready
    torch.testing.assert_close(interpreter.baseline, torch.tensor([1.0]))

    output = interpreter.update(torch.tensor([4.0]))
    assert output is not None
    torch.testing.assert_close(output.state, torch.tensor([3.0, 0.0, 2.0]))


def test_level_trend_and_movement_use_only_past_and_current() -> None:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=1, calibration_steps=2, temporal_window=3
    )
    interpreter.update(torch.tensor([0.0]))
    interpreter.update(torch.tensor([2.0]))
    first = interpreter.update(torch.tensor([4.0]))
    second = interpreter.update(torch.tensor([7.0]))

    assert first is not None and second is not None
    torch.testing.assert_close(second.level, torch.tensor(4.5))
    torch.testing.assert_close(second.trend, torch.tensor(1.5))
    torch.testing.assert_close(second.movement, torch.tensor(3.0))


def test_reset_prevents_cross_episode_state_leakage() -> None:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=1, calibration_steps=2, temporal_window=3
    )
    interpreter.update(torch.tensor([0.0]))
    interpreter.update(torch.tensor([2.0]))
    assert interpreter.update(torch.tensor([4.0])) is not None

    interpreter.reset()
    assert interpreter.phase is InterpreterPhase.CALIBRATING
    assert interpreter.calibration_count == 0
    assert interpreter.baseline is None
    assert interpreter.update(torch.tensor([10.0])) is None
    assert interpreter.update(torch.tensor([14.0])) is None
    torch.testing.assert_close(interpreter.baseline, torch.tensor([12.0]))


def test_baseline_property_does_not_expose_internal_tensor() -> None:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=1, calibration_steps=1
    )
    interpreter.update(torch.tensor([2.0]))
    exposed = interpreter.baseline
    assert exposed is not None
    exposed.add_(100.0)
    torch.testing.assert_close(interpreter.baseline, torch.tensor([2.0]))


@pytest.mark.parametrize(
    "invalid",
    [torch.zeros(1, 2), torch.zeros(3), torch.tensor([float("nan"), 0.0])],
)
def test_rejects_invalid_embedding(invalid: torch.Tensor) -> None:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=2, calibration_steps=2
    )
    with pytest.raises(ValueError):
        interpreter.update(invalid)


def test_rejects_integer_embedding() -> None:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=2, calibration_steps=2
    )
    with pytest.raises(ValueError, match="floating-point"):
        interpreter.update(torch.tensor([1, 2]))


def test_constructor_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        RelativeTemporalStateInterpreter(embedding_dim=0, calibration_steps=2)
    with pytest.raises(ValueError, match="calibration_steps"):
        RelativeTemporalStateInterpreter(embedding_dim=2, calibration_steps=0)
    with pytest.raises(ValueError, match="temporal_window"):
        RelativeTemporalStateInterpreter(
            embedding_dim=2, calibration_steps=2, temporal_window=1
        )
