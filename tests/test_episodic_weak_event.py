"""Synthetic-only checks; no XJTU cache or protected-bearing reads."""

import pytest

from state_interpreter.episodic_weak_event import (
    EpisodeConfig, EpisodePhase, OnlineEpisodicWeakEvent,
)


def make_detector():
    return OnlineEpisodicWeakEvent(EpisodeConfig(provenance="rms_episodic_v2"), bearing_id="synthetic-A")


def calibrate(detector):
    values = [0.0, 2.0] * 7 + [1.0]
    return [detector.update(bearing_id="synthetic-A", step_id=i, value=q) for i, q in enumerate(values)]


def test_short_excursion_never_activates():
    detector = make_detector()
    assert all(x.phase is EpisodePhase.CALIBRATING and x.active_event is None for x in calibrate(detector))
    out = [detector.update(bearing_id="synthetic-A", step_id=i, value=4.0) for i in range(15, 19)]
    assert all(x.active_event is False for x in out)
    assert all(x.candidate_start_step is None for x in out)
    reset = detector.update(bearing_id="synthetic-A", step_id=19, value=1.0)
    assert not reset.entered_now and reset.episode_count == 0


def test_confirm_clear_reconfirm_without_backfill():
    detector = make_detector()
    calibrate(detector)
    first = [detector.update(bearing_id="synthetic-A", step_id=i, value=4.0) for i in range(15, 20)]
    assert all(x.active_event is False for x in first[:4])
    assert first[4].entered_now and first[4].phase is EpisodePhase.ACTIVE
    assert first[4].confirm_step == 19 and first[4].candidate_start_step == 15
    assert first[4].duration_since_confirm == 1
    assert first[0].confirm_step is None  # prior immutable output not rewritten
    below = [detector.update(bearing_id="synthetic-A", step_id=i, value=1.0) for i in range(20, 25)]
    assert all(x.active_event is True for x in below[:4])
    assert below[3].duration_since_confirm == 5
    assert below[4].exited_now and below[4].phase is EpisodePhase.MONITORING
    assert below[4].active_event is False and below[4].episode_count == 1
    second = [detector.update(bearing_id="synthetic-A", step_id=i, value=4.0) for i in range(25, 30)]
    assert second[-1].entered_now and second[-1].episode_count == 2
    assert second[-1].confirm_step == 29 and second[-1].candidate_start_step == 25


def test_gap_interrupts_without_claiming_recovery():
    detector = make_detector()
    calibrate(detector)
    for i in range(15, 20):
        detector.update(bearing_id="synthetic-A", step_id=i, value=4.0)
    gap = detector.update(bearing_id="synthetic-A", step_id=22, value=4.0)
    assert gap.phase is EpisodePhase.UNKNOWN_GAP
    assert gap.active_event is None and gap.interrupted_by_gap
    assert not gap.exited_now and gap.episode_count == 1
    later = [detector.update(bearing_id="synthetic-A", step_id=i, value=4.0) for i in range(23, 28)]
    assert later[-1].entered_now and later[-1].episode_count == 2


def test_continuous_outputs_are_causal_and_prefix_invariant():
    seq = [0.0, 2.0] * 7 + [1.0] + [1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    def run(values):
        detector = make_detector()
        return [detector.update(bearing_id="synthetic-A", step_id=i, value=q) for i, q in enumerate(values)]
    assert run(seq[:19]) == run(seq)[:19]
    output = run(seq)
    assert output[15].deviation_z is not None and output[15].trend == 0.0
    assert output[18].trend is not None and output[18].trend > 0


def test_invalid_reference_and_reset_isolation():
    detector = make_detector()
    for i in range(15):
        detector.update(bearing_id="synthetic-A", step_id=i, value=1.0)
    invalid = detector.update(bearing_id="synthetic-A", step_id=15, value=4.0)
    assert invalid.phase is EpisodePhase.INVALID_REFERENCE and invalid.active_event is None
    with pytest.raises(ValueError, match="reset"):
        detector.update(bearing_id="synthetic-B", step_id=16, value=4.0)
    detector.reset(bearing_id="synthetic-B")
    assert detector.update(bearing_id="synthetic-B", step_id=0, value=1.0).episode_count == 0


def test_nonfinite_and_out_of_order_rejections_are_atomic():
    detector = make_detector()
    calibrate(detector)
    with pytest.raises(ValueError, match="finite"):
        detector.update(bearing_id="synthetic-A", step_id=15, value=float("nan"))
    detector.update(bearing_id="synthetic-A", step_id=15, value=4.0)
    with pytest.raises(ValueError, match="strictly"):
        detector.update(bearing_id="synthetic-A", step_id=15, value=4.0)
    assert detector.update(bearing_id="synthetic-A", step_id=16, value=4.0).step_id == 16


def test_calibration_gap_rejected_without_consuming_observation():
    detector = make_detector()
    detector.update(bearing_id="synthetic-A", step_id=0, value=0.0)
    with pytest.raises(ValueError, match="calibration requires consecutive"):
        detector.update(bearing_id="synthetic-A", step_id=2, value=2.0)
    assert detector.update(bearing_id="synthetic-A", step_id=1, value=2.0).step_id == 1
