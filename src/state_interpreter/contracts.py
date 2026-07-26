"""Data contracts shared by VibFM, datasets, Gearbox, and RL code."""

from dataclasses import dataclass, field
from typing import Mapping

import torch


ScalarAction = int | float


@dataclass(frozen=True)
class RawMeasurement:
    """One ordered vibration measurement before VibFM encoding."""

    episode_id: str
    step_id: int
    time_index: float
    vibration: torch.Tensor
    operating_conditions: Mapping[str, float] = field(default_factory=dict)
    damage_target: float | None = None
    health_index_target: float | None = None
    stage_target: int | None = None
    previous_action: ScalarAction | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if self.vibration.ndim < 1:
            raise ValueError("vibration must have at least one dimension")
        if self.stage_target is not None and self.stage_target < 0:
            raise ValueError("stage_target must be non-negative")


@dataclass(frozen=True)
class EmbeddingSample:
    """One measurement after a fixed VibFM encoder produced ``z_health``."""

    measurement: RawMeasurement
    z_health: torch.Tensor

    def __post_init__(self) -> None:
        if self.z_health.ndim != 1:
            raise ValueError(
                "one EmbeddingSample must contain z_health with shape "
                f"[embedding_dim], got {tuple(self.z_health.shape)}"
            )
        if self.z_health.numel() == 0:
            raise ValueError("z_health must not be empty")


@dataclass(frozen=True)
class StateTransition:
    """One action-conditioned transition for predictive/RL evaluation."""

    current: EmbeddingSample
    action: ScalarAction
    reward: float
    next_sample: EmbeddingSample
    done: bool

    def __post_init__(self) -> None:
        current_measurement = self.current.measurement
        next_measurement = self.next_sample.measurement
        if current_measurement.episode_id != next_measurement.episode_id:
            raise ValueError("current and next samples must share an episode_id")
        if next_measurement.step_id <= current_measurement.step_id:
            raise ValueError("next_sample.step_id must be greater than current step_id")
        if self.current.z_health.shape != self.next_sample.z_health.shape:
            raise ValueError("current and next z_health shapes must match")
