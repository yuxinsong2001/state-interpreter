"""Optional resolution-aware labels over the existing radial interpreter.

No threshold is learned here. A policy is an explicit external declaration,
not a statistical confidence interval or a health classifier.
"""
from dataclasses import dataclass
from enum import Enum
import math
import torch
from state_interpreter.relative_temporal import RelativeTemporalStateInterpreter, RelativeTemporalStateOutput

class DirectionLabel(str, Enum):
    WINDOW_WARMUP = 'window_warmup'
    UNCALIBRATED = 'direction_uncalibrated'
    UNRESOLVED = 'within_resolution'
    AWAY = 'moving_away'
    TOWARD = 'moving_toward'

@dataclass(frozen=True)
class DirectionResolutionPolicy:
    trend_limit: float
    evidence_reference: str
    representation_id: str

    def __post_init__(self):
        if not math.isfinite(self.trend_limit) or self.trend_limit <= 0:
            raise ValueError('trend_limit must be finite and strictly positive')
        if not self.evidence_reference.strip() or not self.representation_id.strip():
            raise ValueError('threshold evidence and representation identity are required')

@dataclass(frozen=True)
class ResolvedRelativeState:
    continuous: RelativeTemporalStateOutput
    direction: DirectionLabel
    trend_limit: float | None
    evidence_reference: str | None

class ResolutionAwareRelativeInterpreter:
    def __init__(self, *, embedding_dim, calibration_steps=15, temporal_window=10,
                 representation_id, policy=None):
        if not representation_id.strip():
            raise ValueError('representation_id required')
        if policy is not None and policy.representation_id != representation_id:
            raise ValueError('policy belongs to another representation')
        self._base = RelativeTemporalStateInterpreter(embedding_dim=embedding_dim,
            calibration_steps=calibration_steps, temporal_window=temporal_window)
        self._window = temporal_window
        self._policy = policy
        self._outputs = 0

    def reset(self):
        self._base.reset()
        self._outputs = 0

    def update(self, z: torch.Tensor):
        value = self._base.update(z)
        if value is None:
            return None
        self._outputs += 1
        if self._outputs < self._window:
            label = DirectionLabel.WINDOW_WARMUP
        elif self._policy is None:
            label = DirectionLabel.UNCALIBRATED
        elif abs(float(value.trend)) <= self._policy.trend_limit:
            label = DirectionLabel.UNRESOLVED
        elif float(value.trend) > 0:
            label = DirectionLabel.AWAY
        else:
            label = DirectionLabel.TOWARD
        return ResolvedRelativeState(value, label,
            None if self._policy is None else self._policy.trend_limit,
            None if self._policy is None else self._policy.evidence_reference)
