"""Explicit reference qualification gate; no automatic health certification."""
from dataclasses import dataclass
from enum import Enum
from state_interpreter.direction_resolution import ResolutionAwareRelativeInterpreter, DirectionLabel


class ReferenceStatus(str, Enum):
    UNREVIEWED = 'unreviewed'
    ACCEPTED = 'accepted_for_relative_description'
    REJECTED = 'rejected'


@dataclass(frozen=True)
class ReferenceQualification:
    status: ReferenceStatus
    episode_id: str
    representation_id: str
    calibration_steps: int
    temporal_window: int
    evidence_reference: str
    rationale: str

    def __post_init__(self):
        if not isinstance(self.status, ReferenceStatus):
            raise ValueError('status must be ReferenceStatus')
        if not all(isinstance(s, str) and s.strip() for s in
                   (self.episode_id, self.representation_id, self.evidence_reference, self.rationale)):
            raise ValueError('scope, evidence and rationale are required')
        if self.calibration_steps < 1 or self.temporal_window < 1:
            raise ValueError('reference settings must be positive')


@dataclass(frozen=True)
class QualifiedRelativeOutput:
    continuous: object
    direction: DirectionLabel
    reference_status: ReferenceStatus
    suppression_reason: str | None


class ReferenceQualifiedInterpreter:
    def __init__(self, *, embedding_dim, episode_id, representation_id,
                 calibration_steps=15, temporal_window=10, policy=None, qualification=None):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('episode_id required')
        if qualification is not None:
            if (qualification.episode_id, qualification.representation_id,
                qualification.calibration_steps, qualification.temporal_window) != (
                    episode_id, representation_id, calibration_steps, temporal_window):
                raise ValueError('qualification scope mismatch')
        self._qualification = qualification
        self._wrapped = ResolutionAwareRelativeInterpreter(embedding_dim=embedding_dim,
            representation_id=representation_id, calibration_steps=calibration_steps,
            temporal_window=temporal_window, policy=policy)

    def reset(self):
        """Replay the SAME episode; a new episode requires a new instance."""
        self._wrapped.reset()

    def update(self, z):
        value = self._wrapped.update(z)
        if value is None:
            return None
        status = ReferenceStatus.UNREVIEWED if self._qualification is None else self._qualification.status
        if status != ReferenceStatus.ACCEPTED:
            return QualifiedRelativeOutput(value.continuous, DirectionLabel.UNCALIBRATED,
                status, 'reference_' + status.value)
        return QualifiedRelativeOutput(value.continuous, value.direction, status, None)
