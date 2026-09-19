"""Development-only weak-event evaluation on already materialized feature arrays.

The output is an algorithmic candidate event, never a verified fault onset.
"""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite

import numpy as np

from state_interpreter.features.bearing_features import FEATURE_NAMES
from state_interpreter.weak_event_detector import (
    EventPhase,
    OnlineWeakEventDetector,
    WeakEventConfig,
    pair_mean,
)
from state_interpreter.weak_event_protocol import DEVELOPMENT, PAIRS


def evaluate_bearing(
    bearing_id: str,
    features: np.ndarray,
    step_ids: np.ndarray,
    *,
    calibration_steps: int = 15,
    sigma_multiplier: float = 2.0,
    consecutive_exceedances: int = 5,
    min_reference_std: float = 1e-8,
) -> tuple[list[dict], list[dict]]:
    """Evaluate both prespecified channels; validate the entire run first.

    Missing step IDs are allowed but break consecutive exceedance runs.
    Nonfinite values, duplicate/out-of-order IDs, and missing rows fail closed.
    """
    if bearing_id not in DEVELOPMENT:
        raise PermissionError("protected or unknown bearing")
    values = np.asarray(features)
    steps = np.asarray(step_ids)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("expected [measurement, 65] features")
    if steps.ndim != 1 or len(steps) != len(values) or len(steps) < calibration_steps:
        raise ValueError("invalid step count")
    if values.dtype.kind not in "fi" or not np.isfinite(values).all():
        raise ValueError("features must be finite numeric values")
    if steps.dtype.kind not in "iu" or (steps < 0).any() or (np.diff(steps) <= 0).any():
        raise ValueError("step IDs must be nonnegative strictly increasing integers")

    trajectories: list[dict] = []
    summaries: list[dict] = []
    for provenance, names in PAIRS.items():
        detector = OnlineWeakEventDetector(
            WeakEventConfig(
                provenance=provenance,
                calibration_steps=calibration_steps,
                sigma_multiplier=sigma_multiplier,
                consecutive_exceedances=consecutive_exceedances,
                min_reference_std=min_reference_std,
            ),
            bearing_id=bearing_id,
        )
        indices = [FEATURE_NAMES.index(name) for name in names]
        channel_rows: list[dict] = []
        for position, (step, vector) in enumerate(zip(steps, values)):
            score = pair_mean(float(vector[indices[0]]), float(vector[indices[1]]))
            output = detector.update(bearing_id=bearing_id, step_id=int(step), value=score)
            row = asdict(output)
            row["phase"] = output.phase.value
            row["position"] = position
            # Defensive check before serializing CSV/JSON.
            if output.event_strength is not None and not isfinite(output.event_strength):
                raise ValueError("nonfinite event strength")
            channel_rows.append(row)
        trajectories.extend(channel_rows)
        alarm_positions = [row["position"] for row in channel_rows if row["event_confirm_step"] == row["step_id"]]
        if len(alarm_positions) > 1:
            raise AssertionError("event confirmed more than once")
        confirm_position = alarm_positions[0] if alarm_positions else None
        last = channel_rows[-1]
        summaries.append({
            "bearing_id": bearing_id,
            "provenance": provenance,
            "measurement_count": len(channel_rows),
            "first_step": int(steps[0]),
            "recorded_end_step": int(steps[-1]),
            "reference_valid": last["threshold"] is not None,
            "reference_mean": last["reference_mean"],
            "reference_std": last["reference_std"],
            "threshold": last["threshold"],
            "gap_count": sum(bool(row["gap_before_step"]) for row in channel_rows),
            "event_confirm_step": last["event_confirm_step"],
            "candidate_start_step": last["candidate_start_step"],
            "event_fraction": None if confirm_position is None else confirm_position / (len(channel_rows) - 1),
            "lead_to_recorded_end_measurements": None if confirm_position is None else len(channel_rows) - 1 - confirm_position,
            "interpretation": "algorithmic candidate event, not verified fault onset",
        })
    return trajectories, summaries
