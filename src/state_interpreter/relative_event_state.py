"""Interpretable event metadata over the frozen causal v2 detector."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from state_interpreter.episodic_weak_event import (
    EpisodeConfig, EpisodeOutput, EpisodePhase, OnlineEpisodicWeakEvent,
)


@dataclass(frozen=True)
class EpisodeTermination:
    episode_id: int
    candidate_start_step: int
    confirm_step: int
    last_observed_active_step: int
    termination_observed_step: int
    observed_active_measurements: int
    reason: str
    right_censored: bool


@dataclass(frozen=True)
class RelativeEventState:
    schema_version: str
    observation: EpisodeOutput
    feature_deviation_z: float | None
    feature_trend_z_per_measurement: float | None
    trend_observation_count: int
    transition: str
    termination_now: EpisodeTermination | None
    last_termination: EpisodeTermination | None

    def to_dict(self) -> dict:
        result = asdict(self)
        result['observation']['phase'] = self.observation.phase.value
        return result


class RelativeEventStateInterpreter:
    """One causal feature stream; an event exit never denotes physical recovery.

    Continuous deviation and trend coexist with the discrete event state.
    No health-stage, damage-percentage, or latent Movement interpretation is added.
    """

    def __init__(self, config: EpisodeConfig, *, bearing_id: str):
        self.config = config
        self._detector = OnlineEpisodicWeakEvent(config, bearing_id=bearing_id)
        self.reset(bearing_id=bearing_id)

    def reset(self, *, bearing_id: str) -> None:
        self._detector.reset(bearing_id=bearing_id)
        self._previous: EpisodeOutput | None = None
        self._last_termination: EpisodeTermination | None = None
        self._trend_count = 0
        self._closed = False

    def _terminate(self, step: int, reason: str, censored: bool) -> EpisodeTermination:
        previous = self._previous
        assert previous is not None and previous.active_event
        assert previous.active_episode_id is not None
        assert previous.candidate_start_step is not None and previous.confirm_step is not None
        assert previous.duration_since_confirm is not None
        return EpisodeTermination(
            previous.active_episode_id, previous.candidate_start_step, previous.confirm_step,
            previous.step_id, step, previous.duration_since_confirm, reason, censored,
        )

    def update(self, *, bearing_id: str, step_id: int, value: float) -> RelativeEventState:
        if self._closed:
            raise ValueError('record finished; reset before updating')
        # Delegate first: a rejected input does not alter wrapper metadata.
        out = self._detector.update(bearing_id=bearing_id, step_id=step_id, value=value)
        termination = None
        if out.interrupted_by_gap:
            termination = self._terminate(step_id, 'measurement_gap', True)
        elif out.exited_now:
            termination = self._terminate(step_id, 'consecutive_non_exceedances', False)
        if termination is not None:
            self._last_termination = termination
        if out.phase in (EpisodePhase.CALIBRATING, EpisodePhase.INVALID_REFERENCE):
            self._trend_count = 0
        elif out.phase is EpisodePhase.UNKNOWN_GAP:
            self._trend_count = 1
        else:
            self._trend_count = min(self.config.trend_window, self._trend_count + 1)
        transition = ('interrupted' if out.interrupted_by_gap else
                      'exited' if out.exited_now else 'entered' if out.entered_now else 'none')
        state = RelativeEventState(
            'relative_event_state_v1', out, out.deviation_z,
            out.trend if self._trend_count >= 2 else None,
            self._trend_count, transition, termination, self._last_termination,
        )
        self._previous = out
        return state

    def finish_record(self) -> EpisodeTermination | None:
        """Explicit end-of-record metadata; never alters past online outputs."""
        if self._closed:
            raise ValueError('record already finished')
        self._closed = True
        if self._previous is not None and self._previous.active_event:
            self._last_termination = self._terminate(self._previous.step_id, 'record_end', True)
            return self._last_termination
        return None
