"""Causal, resettable candidate-change episodes; not physical fault labels."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt


class EpisodePhase(str, Enum):
    CALIBRATING = "CALIBRATING"
    MONITORING = "MONITORING"
    ACTIVE = "ACTIVE"
    UNKNOWN_GAP = "UNKNOWN_GAP"
    INVALID_REFERENCE = "INVALID_REFERENCE"


@dataclass(frozen=True)
class EpisodeConfig:
    provenance: str
    calibration_steps: int = 15
    sigma_multiplier: float = 2.0
    enter_steps: int = 5
    exit_steps: int = 5
    trend_window: int = 10
    min_reference_std: float = 1e-8

    def __post_init__(self) -> None:
        if self.provenance not in {"rms_episodic_v2", "kurtosis_episodic_v2"}:
            raise ValueError("unknown provenance")
        if min(self.calibration_steps, self.enter_steps, self.exit_steps) < 1:
            raise ValueError("step counts must be positive")
        if self.trend_window < 2:
            raise ValueError("trend_window must be at least two")
        if not isfinite(self.sigma_multiplier) or self.sigma_multiplier < 0:
            raise ValueError("invalid sigma multiplier")
        if not isfinite(self.min_reference_std) or self.min_reference_std <= 0:
            raise ValueError("invalid minimum reference std")


@dataclass(frozen=True)
class EpisodeOutput:
    bearing_id: str
    step_id: int
    phase: EpisodePhase
    active_event: bool | None
    episode_count: int
    active_episode_id: int | None
    entered_now: bool
    exited_now: bool
    interrupted_by_gap: bool
    confirm_step: int | None
    candidate_start_step: int | None
    duration_since_confirm: int | None
    deviation_z: float | None
    trend: float | None
    value: float
    reference_mean: float | None
    reference_std: float | None
    threshold: float | None
    provenance: str


class OnlineEpisodicWeakEvent:
    """One instance per bearing; reset explicitly before another bearing."""

    def __init__(self, config: EpisodeConfig, *, bearing_id: str) -> None:
        self.config = config
        self.reset(bearing_id=bearing_id)

    def reset(self, *, bearing_id: str) -> None:
        if not isinstance(bearing_id, str) or not bearing_id:
            raise ValueError("bearing_id must be a nonempty string")
        self.bearing_id = bearing_id
        self._last_step: int | None = None
        self._calibration: list[float] = []
        self._phase = EpisodePhase.CALIBRATING
        self._mean: float | None = None
        self._std: float | None = None
        self._threshold: float | None = None
        self._above_count = 0
        self._below_count = 0
        self._candidate_start: int | None = None
        self._confirm_step: int | None = None
        self._episode_count = 0
        self._trend_values: deque[float] = deque(maxlen=self.config.trend_window)

    def update(self, *, bearing_id: str, step_id: int, value: float) -> EpisodeOutput:
        # Reject invalid calls without changing the stream state.
        if bearing_id != self.bearing_id:
            raise ValueError("bearing changed without reset")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
            raise ValueError("step_id must be a nonnegative integer")
        if self._last_step is not None and step_id <= self._last_step:
            raise ValueError("step_id must increase strictly")
        q = float(value)
        if not isfinite(q):
            raise ValueError("value must be finite")

        gap = self._last_step is not None and step_id != self._last_step + 1
        if self._phase is EpisodePhase.CALIBRATING:
            if gap:
                raise ValueError("calibration requires consecutive step IDs")
            self._last_step = step_id
            self._calibration.append(q)
            if len(self._calibration) == self.config.calibration_steps:
                mean = sum(self._calibration) / len(self._calibration)
                std = sqrt(sum((x - mean) ** 2 for x in self._calibration) / len(self._calibration))
                threshold = mean + self.config.sigma_multiplier * std
                self._mean, self._std = mean, std
                if not all(map(isfinite, (mean, std, threshold))) or std < self.config.min_reference_std:
                    self._phase = EpisodePhase.INVALID_REFERENCE
                else:
                    self._threshold = threshold
                    self._phase = EpisodePhase.MONITORING
            return self._output(step_id, q, EpisodePhase.CALIBRATING, None, None)

        if self._phase is EpisodePhase.INVALID_REFERENCE:
            self._last_step = step_id
            return self._output(step_id, q, EpisodePhase.INVALID_REFERENCE, None, None)
        assert self._mean is not None and self._std is not None and self._threshold is not None
        deviation = (q - self._mean) / self._std
        if not isfinite(deviation):
            raise ValueError("nonfinite normalized deviation")
        self._last_step = step_id
        if gap:
            # Continuity is unknown: interrupt, do not claim a healthy recovery.
            was_active = self._phase is EpisodePhase.ACTIVE
            self._phase = EpisodePhase.MONITORING
            self._above_count = self._below_count = 0
            self._candidate_start = self._confirm_step = None
            self._trend_values.clear()
            self._trend_values.append(deviation)
            return self._output(
                step_id, q, EpisodePhase.UNKNOWN_GAP, deviation, 0.0,
                interrupted_by_gap=was_active,
            )

        self._trend_values.append(deviation)
        trend = self._slope(self._trend_values)
        entered = exited = False
        if self._phase is EpisodePhase.MONITORING:
            if q > self._threshold:
                if self._above_count == 0:
                    self._candidate_start = step_id
                self._above_count += 1
                if self._above_count == self.config.enter_steps:
                    self._episode_count += 1
                    self._confirm_step = step_id
                    self._phase = EpisodePhase.ACTIVE
                    self._below_count = 0
                    entered = True
            else:
                self._above_count = 0
                self._candidate_start = None
        elif self._phase is EpisodePhase.ACTIVE:
            if q <= self._threshold:
                self._below_count += 1
                if self._below_count == self.config.exit_steps:
                    self._phase = EpisodePhase.MONITORING
                    self._above_count = self._below_count = 0
                    self._candidate_start = self._confirm_step = None
                    exited = True
            else:
                self._below_count = 0
        return self._output(step_id, q, self._phase, deviation, trend, entered, exited)

    @staticmethod
    def _slope(values: deque[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        mean_x = (n - 1) / 2
        mean_y = sum(values) / n
        return sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values)) / sum(
            (i - mean_x) ** 2 for i in range(n)
        )

    def _output(
        self, step_id: int, q: float, phase: EpisodePhase,
        deviation: float | None, trend: float | None,
        entered: bool = False, exited: bool = False,
        interrupted_by_gap: bool = False,
    ) -> EpisodeOutput:
        active = None if phase in (EpisodePhase.CALIBRATING, EpisodePhase.UNKNOWN_GAP, EpisodePhase.INVALID_REFERENCE) else phase is EpisodePhase.ACTIVE
        return EpisodeOutput(
            bearing_id=self.bearing_id, step_id=step_id, phase=phase,
            active_event=active, episode_count=self._episode_count,
            active_episode_id=self._episode_count if active else None,
            entered_now=entered, exited_now=exited,
            interrupted_by_gap=interrupted_by_gap,
            confirm_step=self._confirm_step if active else None,
            candidate_start_step=self._candidate_start if active else None,
            duration_since_confirm=step_id - self._confirm_step + 1 if active and self._confirm_step is not None else None,
            deviation_z=deviation, trend=trend, value=q,
            reference_mean=self._mean, reference_std=self._std,
            threshold=self._threshold, provenance=self.config.provenance,
        )
