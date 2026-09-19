"""Synthetic-only integration checks for the Condition 3 evaluator."""

import numpy as np
import pytest

from state_interpreter.features.bearing_features import FEATURE_NAMES
from state_interpreter.weak_event_development import evaluate_bearing


def synthetic_features(n=21):
    features = np.ones((n, len(FEATURE_NAMES)), dtype=float)
    reference = [0.0, 2.0] * 7 + [1.0]
    for name in ("h_rms", "v_rms", "h_kurtosis", "v_kurtosis"):
        features[:15, FEATURE_NAMES.index(name)] = reference
    for name in ("h_rms", "v_rms"):
        features[15:20, FEATURE_NAMES.index(name)] = 4.0
    return features


def test_both_branches_reported_without_selecting_winner():
    rows, summaries = evaluate_bearing("Bearing3_1", synthetic_features(), np.arange(21))
    assert len(rows) == 42 and len(summaries) == 2
    rms, kurtosis = summaries
    assert rms["provenance"] == "rms_threshold_v1"
    assert rms["event_confirm_step"] == 19
    assert rms["candidate_start_step"] == 15
    assert rms["lead_to_recorded_end_measurements"] == 1
    assert kurtosis["provenance"] == "kurtosis_threshold_v1"
    assert kurtosis["event_confirm_step"] is None
    early = [x for x in rows if x["provenance"] == "rms_threshold_v1" and x["step_id"] == 15][0]
    assert early["event_flag"] == 0 and early["candidate_start_step"] is None


def test_gap_resets_run_and_position_fraction_is_posthoc():
    steps = np.arange(21)
    steps[19:] += 1
    _, summaries = evaluate_bearing("Bearing3_2", synthetic_features(), steps)
    assert summaries[0]["gap_count"] == 1
    assert summaries[0]["event_confirm_step"] is None


def test_bad_inputs_fail_closed():
    features = synthetic_features()
    with pytest.raises(PermissionError):
        evaluate_bearing("Bearing3_4", features, np.arange(21))
    bad = features.copy()
    bad[18, FEATURE_NAMES.index("h_rms")] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_bearing("Bearing3_1", bad, np.arange(21))
    steps = np.arange(21)
    steps[18] = 17
    with pytest.raises(ValueError, match="strictly"):
        evaluate_bearing("Bearing3_1", features, steps)


def test_constant_reference_is_not_valid_or_forced_alarm():
    features = np.ones((21, len(FEATURE_NAMES)))
    _, summaries = evaluate_bearing("Bearing3_3", features, np.arange(21))
    assert len(summaries) == 2
    assert all(not item["reference_valid"] and item["event_confirm_step"] is None for item in summaries)
