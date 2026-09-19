"""Explicitly gated Condition 3 development runner; never accepts bearing paths."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.weak_event_development import evaluate_bearing
from state_interpreter.weak_event_execution_lock import verify_execution_lock
from state_interpreter.weak_event_protocol import WeakEventDevelopmentPolicy, file_sha256

CONFIG_PATH = ROOT / "configs/xjtu_condition3_weak_event_development_v1.json"
LOCK_PATH = ROOT / "configs/xjtu_condition3_weak_event_execution_lock_v1.json"


def _load_development_arrays(policy: WeakEventDevelopmentPolicy):
    """This function is reachable only after policy authorization + hash preflight."""
    cache = ROOT / policy.config["input"]["cache_directory"]
    for entry in policy.config["input"]["cache_entries"]:
        bearing = entry["bearing_id"]
        path = cache / entry["path"]
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(policy.config["input"]["required_fields"]):
                raise ValueError(f"unexpected NPZ fields: {bearing}")
            features = np.asarray(payload["features"])
            step_ids = np.asarray(payload["step_ids"])
        if len(features) != entry["sample_count"]:
            raise ValueError(f"sample count mismatch: {bearing}")
        yield bearing, features, step_ids


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--execute-development", action="store_true")
    args = parser.parse_args()

    verified_execution_code = verify_execution_lock(ROOT, LOCK_PATH)
    policy = WeakEventDevelopmentPolicy.load(CONFIG_PATH, ROOT)
    policy.authorize(["Bearing3_1", "Bearing3_2", "Bearing3_3"])
    preflight = policy.verify_metadata_and_hashes()
    preflight["verified_execution_code"] = verified_execution_code
    if args.check_only:
        print(json.dumps(preflight, indent=2))
        return

    detector = policy.config["detector"]
    all_rows: list[dict] = []
    all_summaries: list[dict] = []
    for bearing, features, step_ids in _load_development_arrays(policy):
        rows, summaries = evaluate_bearing(
            bearing, features, step_ids,
            calibration_steps=detector["calibration_steps"],
            sigma_multiplier=detector["sigma_multiplier"],
            consecutive_exceedances=detector["consecutive_exceedances"],
            min_reference_std=detector["min_reference_std"],
        )
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    output = ROOT / policy.config["output_directory"]
    record = ROOT / policy.config["record_path"]
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists() or record.exists():
        raise FileExistsError("weak-event output/record already exists")
    report = {
        "status": "development_completed_candidate_events_only",
        "experiment_id": policy.config["experiment_id"],
        "config_sha256": file_sha256(CONFIG_PATH),
        "preflight": preflight,
        "summaries": all_summaries,
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "physical_onset_accuracy": None,
    }
    staging.mkdir(parents=True)
    _write_csv(staging / "candidate_event_trajectories.csv", all_rows)
    _write_csv(staging / "candidate_event_summary.csv", all_summaries)
    (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging.rename(output)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"status": report["status"], "config_sha256": report["config_sha256"],
                                  "result_report_sha256": file_sha256(output / "report.json"),
                                  "protected_bearings_not_read": report["protected_bearings_not_read"]}, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "summary_rows": len(all_summaries)}, indent=2))


if __name__ == "__main__":
    main()
