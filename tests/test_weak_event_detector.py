"""Synthetic-only tests: never access XJTU measurements."""

import pytest

from state_interpreter.weak_event_detector import (
    EventPhase, OnlineWeakEventDetector, WeakEventConfig, pair_mean,
)


def detector(bearing="synthetic-A", provenance="rms_threshold_v1"):
    return OnlineWeakEventDetector(WeakEventConfig(provenance=provenance), bearing_id=bearing)


def calibrate(d, bearing="synthetic-A"):
    # Population reference: mean=1, std=sqrt(14/15), threshold < 3.
    out = [d.update(bearing_id=bearing, step_id=i, value=v)
           for i, v in enumerate([0.0, 2.0] * 7 + [1.0])]
    assert all(x.phase is EventPhase.CALIBRATING and x.event_flag == 0 for x in out)


def test_fifth_exceedance_confirms_without_backfill_and_latches():
    d = detector()
    calibrate(d)
    first = [d.update(bearing_id="synthetic-A", step_id=i, value=4.0) for i in range(15, 20)]
    assert all(x.event_flag == 0 and x.candidate_start_step is None for x in first[:4])
    assert first[4].phase is EventPhase.ALARMED
    assert first[4].event_confirm_step == 19
    assert first[4].candidate_start_step == 15
    assert first[0].event_flag == 0  # immutable earlier output
    later = d.update(bearing_id="synthetic-A", step_id=20, value=0.0)
    assert later.event_flag == 1 and later.event_confirm_step == 19


def test_gap_and_equality_break_consecutive_run():
    d = detector()
    calibrate(d)
    for i in range(15, 19):
        d.update(bearing_id="synthetic-A", step_id=i, value=4.0)
    gap = d.update(bearing_id="synthetic-A", step_id=20, value=4.0)
    assert gap.gap_before_step and gap.event_flag == 0
    threshold = gap.threshold
    equal = d.update(bearing_id="synthetic-A", step_id=21, value=threshold)
    assert equal.event_flag == 0
    result = [d.update(bearing_id="synthetic-A", step_id=i, value=4.0) for i in range(22, 27)]
    assert result[-1].event_confirm_step == 26 and result[-1].candidate_start_step == 22


def test_prefix_invariance_and_no_alarm():
    values = [0.0, 2.0] * 7 + [1.0] + [1.0] * 8 + [4.0] * 5
    def run(seq):
        d = detector()
        return [d.update(bearing_id="synthetic-A", step_id=i, value=v) for i, v in enumerate(seq)]
    assert run(values[:20]) == run(values)[:20]
    assert all(x.event_flag == 0 for x in run(values[:23]))


def test_invalid_reference_no_event_and_reset_isolated():
    d = detector()
    for i in range(15):
        d.update(bearing_id="synthetic-A", step_id=i, value=1.0)
    x = d.update(bearing_id="synthetic-A", step_id=15, value=100.0)
    assert x.phase is EventPhase.INVALID_REFERENCE and x.event_strength is None and x.event_flag == 0
    with pytest.raises(ValueError, match="reset"):
        d.update(bearing_id="synthetic-B", step_id=16, value=100.0)
    d.reset(bearing_id="synthetic-B")
    assert d.update(bearing_id="synthetic-B", step_id=0, value=0.0).phase is EventPhase.CALIBRATING


def test_rejected_inputs_do_not_mutate_and_comparators_separate():
    d = detector(provenance="kurtosis_threshold_v1")
    with pytest.raises(ValueError):
        d.update(bearing_id="synthetic-A", step_id=0, value=float("nan"))
    assert d.update(bearing_id="synthetic-A", step_id=0, value=pair_mean(0, 2)).provenance == "kurtosis_threshold_v1"
    with pytest.raises(ValueError):
        d.update(bearing_id="synthetic-A", step_id=0, value=1)
    with pytest.raises(ValueError):
        pair_mean(float("inf"), 1)
    assert d.update(bearing_id="synthetic-A", step_id=1, value=1).step_id == 1
