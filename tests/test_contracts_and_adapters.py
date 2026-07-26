import torch

from state_interpreter.adapters import VibFMAdapter
from state_interpreter.contracts import (
    EmbeddingSample,
    RawMeasurement,
    StateTransition,
)


class FakeVibFMAdapter(VibFMAdapter):
    @property
    def embedding_dim(self) -> int:
        return 256

    def encode(self, vibration: torch.Tensor) -> torch.Tensor:
        mean = vibration.float().mean()
        return mean.repeat(self.embedding_dim)


def make_measurement(step_id: int) -> RawMeasurement:
    return RawMeasurement(
        episode_id="gearbox-run-001",
        step_id=step_id,
        time_index=float(step_id * 128),
        vibration=torch.arange(1024, dtype=torch.float32),
        operating_conditions={"torque": 200.0, "rotational_frequency": 30.0},
        damage_target=0.1 * step_id,
        previous_action=0,
    )


def test_vibfm_adapter_preserves_measurement_metadata() -> None:
    measurement = make_measurement(step_id=0)

    sample = FakeVibFMAdapter().transform(measurement)

    assert sample.z_health.shape == (256,)
    assert sample.measurement is measurement
    assert sample.measurement.operating_conditions["torque"] == 200.0


def test_transition_connects_consecutive_samples() -> None:
    adapter = FakeVibFMAdapter()
    current = adapter.transform(make_measurement(step_id=0))
    next_sample = adapter.transform(make_measurement(step_id=1))

    transition = StateTransition(
        current=current,
        action=1,
        reward=0.5,
        next_sample=next_sample,
        done=False,
    )

    assert transition.current.z_health.shape == transition.next_sample.z_health.shape
    assert transition.next_sample.measurement.time_index == 128.0


def test_embedding_sample_rejects_batched_embedding() -> None:
    try:
        EmbeddingSample(
            measurement=make_measurement(step_id=0),
            z_health=torch.zeros(1, 256),
        )
    except ValueError as exc:
        assert "shape [embedding_dim]" in str(exc)
    else:
        raise AssertionError("a batched embedding must be rejected")


def test_transition_rejects_cross_episode_pair() -> None:
    adapter = FakeVibFMAdapter()
    current = adapter.transform(make_measurement(step_id=0))
    other_episode = RawMeasurement(
        episode_id="gearbox-run-002",
        step_id=1,
        time_index=128.0,
        vibration=torch.ones(1024),
    )
    next_sample = adapter.transform(other_episode)

    try:
        StateTransition(
            current=current,
            action=0,
            reward=0.0,
            next_sample=next_sample,
            done=True,
        )
    except ValueError as exc:
        assert "share an episode_id" in str(exc)
    else:
        raise AssertionError("cross-episode transitions must be rejected")
