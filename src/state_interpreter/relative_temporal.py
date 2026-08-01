"""Online, calibration-based State Interpreter for ordered embeddings."""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import NamedTuple

import torch


class InterpreterPhase(str, Enum):
    """Lifecycle of one episode-specific interpreter instance."""

    CALIBRATING = "calibrating"
    READY = "ready"


class RelativeTemporalStateOutput(NamedTuple):
    """Three interpretable state components and their packed vector."""

    level: torch.Tensor
    trend: torch.Tensor
    movement: torch.Tensor
    state: torch.Tensor


class RelativeTemporalStateInterpreter:
    """Interpret one ordered embedding stream relative to its early baseline.

    The first ``calibration_steps`` embeddings define an episode-specific
    baseline. Those calls return ``None``. Subsequent calls use current and past
    values only and return ``[level, trend, movement]``.
    """

    state_dim = 3

    def __init__(
        self,
        *,
        embedding_dim: int,
        calibration_steps: int,
        temporal_window: int = 5,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if calibration_steps <= 0:
            raise ValueError("calibration_steps must be positive")
        if temporal_window <= 1:
            raise ValueError("temporal_window must be greater than one")
        self.embedding_dim = embedding_dim
        self.calibration_steps = calibration_steps
        self.temporal_window = temporal_window
        self.reset()

    def reset(self) -> None:
        """Discard all state before starting another episode."""

        self._phase = InterpreterPhase.CALIBRATING
        self._calibration: list[torch.Tensor] = []
        self._baseline: torch.Tensor | None = None
        self._previous_z: torch.Tensor | None = None
        self._distance_history: deque[torch.Tensor] = deque(
            maxlen=self.temporal_window
        )
        self._level_history: deque[torch.Tensor] = deque(
            maxlen=self.temporal_window
        )

    @property
    def phase(self) -> InterpreterPhase:
        return self._phase

    @property
    def ready(self) -> bool:
        return self._phase is InterpreterPhase.READY

    @property
    def calibration_count(self) -> int:
        return len(self._calibration)

    @property
    def baseline(self) -> torch.Tensor | None:
        """Return a defensive copy of the calibrated baseline."""

        return None if self._baseline is None else self._baseline.clone()

    def _validate(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 1 or z.numel() != self.embedding_dim:
            raise ValueError(
                f"expected z with shape [{self.embedding_dim}], got {tuple(z.shape)}"
            )
        if not torch.is_floating_point(z):
            raise ValueError("z must be a floating-point tensor")
        if not bool(torch.isfinite(z).all()):
            raise ValueError("z contains NaN or infinite values")
        return z.detach().clone()

    @staticmethod
    def _slope(values: deque[torch.Tensor]) -> torch.Tensor:
        if len(values) < 2:
            return torch.zeros_like(values[-1])
        y = torch.stack(tuple(values))
        x = torch.arange(len(y), device=y.device, dtype=y.dtype)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        return (x_centered * y_centered).sum() / x_centered.square().sum()

    def update(self, z: torch.Tensor) -> RelativeTemporalStateOutput | None:
        """Consume one chronological embedding and possibly emit a state."""

        current = self._validate(z)
        if self._phase is InterpreterPhase.CALIBRATING:
            self._calibration.append(current)
            self._previous_z = current
            if len(self._calibration) == self.calibration_steps:
                self._baseline = torch.stack(self._calibration).mean(dim=0)
                self._phase = InterpreterPhase.READY
            return None

        assert self._baseline is not None
        assert self._previous_z is not None
        level_raw = torch.linalg.vector_norm(current - self._baseline)
        movement = torch.linalg.vector_norm(current - self._previous_z)
        self._distance_history.append(level_raw)
        level = torch.stack(tuple(self._distance_history)).mean()
        self._level_history.append(level)
        trend = self._slope(self._level_history)
        state = torch.stack((level, trend, movement))
        self._previous_z = current
        return RelativeTemporalStateOutput(
            level=level,
            trend=trend,
            movement=movement,
            state=state,
        )
