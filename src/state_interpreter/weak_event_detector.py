"""Causal, per-bearing candidate-event detector (not a fault-onset label)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt


class EventPhase(str, Enum):
    CALIBRATING = "CALIBRATING"
    MONITORING = "MONITORING"
    ALARMED = "ALARMED"
    INVALID_REFERENCE = "INVALID_REFERENCE"


@dataclass(frozen=True)
class WeakEventConfig:
    provenance: str
    calibration_steps: int = 15
    sigma_multiplier: float = 2.0
    consecutive_exceedances: int = 5
    min_reference_std: float = 1e-8

    def __post_init__(self) -> None:
        if self.provenance not in {"rms_threshold_v1", "kurtosis_threshold_v1"}:
            raise ValueError("unknown weak-event provenance")
        if self.calibration_steps < 1 or self.consecutive_exceedances < 1:
            raise ValueError("step counts must be positive")
        if not isfinite(self.sigma_multiplier) or self.sigma_multiplier < 0:
            raise ValueError("sigma_multiplier must be finite and nonnegative")
        if not isfinite(self.min_reference_std) or self.min_reference_std <= 0:
            raise ValueError("min_reference_std must be finite and positive")


@dataclass(frozen=True)
class WeakEventOutput:
    bearing_id: str
    step_id: int
    phase: EventPhase
    event_flag: int
    event_confirm_step: int | None
    # Only populated after confirmation; never backfills earlier outputs.
    candidate_start_step: int | None
    event_strength: float | None
    value: float
    reference_mean: float | None
    reference_std: float | None
    threshold: float | None
    provenance: str
    gap_before_step: bool


def pair_mean(horizontal: float, vertical: float) -> float:
    """Combine one horizontal/vertical feature pair without reading data files."""
    h, v = float(horizontal), float(vertical)
    if not isfinite(h) or not isfinite(v):
        raise ValueError("feature values must be finite")
    result = (h + v) / 2.0
    if not isfinite(result):
        raise ValueError("pair mean is nonfinite")
    return result


class OnlineWeakEventDetector:
    """One instance handles one bearing at a time; reset explicitly between runs."""

    def __init__(self, config: WeakEventConfig, *, bearing_id: str) -> None:
        self.config = config
        self.reset(bearing_id=bearing_id)

    def reset(self, *, bearing_id: str) -> None:
        if not isinstance(bearing_id, str) or not bearing_id:
            raise ValueError("bearing_id must be a nonempty string")
        self.bearing_id = bearing_id
        self._last_step: int | None = None
        self._calibration: list[float] = []
        self._phase = EventPhase.CALIBRATING
        self._mean: float | None = None
        self._std: float | None = None
        self._threshold: float | None = None
        self._run_length = 0
        self._run_start: int | None = None
        self._confirm_step: int | None = None
        self._candidate_start: int | None = None

    def update(self, *, bearing_id: str, step_id: int, value: float) -> WeakEventOutput:
        # Validate before mutation so an invalid call can be retried safely.
        if bearing_id != self.bearing_id:
            raise ValueError("bearing changed without explicit reset")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
            raise ValueError("step_id must be a nonnegative integer")
        if self._last_step is not None and step_id <= self._last_step:
            raise ValueError("step_id must increase strictly")
        q = float(value)
        if not isfinite(q):
            raise ValueError("value must be finite")

        gap = self._last_step is not None and step_id != self._last_step + 1
        self._last_step = step_id
        if self._phase is EventPhase.CALIBRATING:
            self._calibration.append(q)
            if len(self._calibration) == self.config.calibration_steps:
                mean = sum(self._calibration) / len(self._calibration)
                std = sqrt(sum((x - mean) ** 2 for x in self._calibration) / len(self._calibration))
                threshold = mean + self.config.sigma_multiplier * std
                self._mean, self._std = mean, std
                if not all(map(isfinite, (mean, std, threshold))) or std < self.config.min_reference_std:
                    self._phase = EventPhase.INVALID_REFERENCE
                else:
                    self._threshold = threshold
                    self._phase = EventPhase.MONITORING
            # Calibration endpoint does not produce an alarm or strength.
            return self._output(step_id, q, EventPhase.CALIBRATING, gap, None)

        if self._phase is EventPhase.INVALID_REFERENCE:
            return self._output(step_id, q, self._phase, gap, None)
        assert self._mean is not None and self._std is not None and self._threshold is not None
        if gap:
            self._run_length, self._run_start = 0, None
        if self._phase is EventPhase.MONITORING:
            if q > self._threshold:
                if self._run_length == 0:
                    self._run_start = step_id
                self._run_length += 1
                if self._run_length == self.config.consecutive_exceedances:
                    self._confirm_step = step_id
                    self._candidate_start = self._run_start
                    self._phase = EventPhase.ALARMED
            else:
                self._run_length, self._run_start = 0, None
        strength = (q - self._mean) / max(self._std, self.config.min_reference_std)
        return self._output(step_id, q, self._phase, gap, strength)

    def _output(self, step_id: int, q: float, phase: EventPhase, gap: bool, strength: float | None) -> WeakEventOutput:
        return WeakEventOutput(
            bearing_id=self.bearing_id, step_id=step_id, phase=phase,
            event_flag=int(self._confirm_step is not None),
            event_confirm_step=self._confirm_step,
            candidate_start_step=self._candidate_start,
            event_strength=strength, value=q, reference_mean=self._mean,
            reference_std=self._std, threshold=self._threshold,
            provenance=self.config.provenance, gap_before_step=gap,
        )
