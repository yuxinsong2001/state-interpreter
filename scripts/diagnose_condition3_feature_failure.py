"""Diagnose Feature LSTM failure using only preregistered development caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler
from state_interpreter.features.bearing_features import FEATURE_NAMES
from state_interpreter.representation_diagnostics import (
    evenly_spaced_indices,
    feature_lifetime_spearman,
    nearest_centroid_predict,
    ridge_fit_predict,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median_feature_autocorrelation(values: np.ndarray, lag: int) -> float:
    correlations = []
    for feature in range(values.shape[1]):
        left = values[:-lag, feature]
        right = values[lag:, feature]
        if left.std() < 1e-12 or right.std() < 1e-12:
            continue
        correlation = np.corrcoef(left, right)[0, 1]
        if np.isfinite(correlation):
            correlations.append(correlation)
    return float(np.median(correlations)) if correlations else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_development_only_diagnosis":
        raise ValueError("invalid failure diagnosis status")
    if args.confirm != config["execution_token"]:
        raise PermissionError("exact diagnosis token required")
    bearings = ["Bearing3_1", "Bearing3_2", "Bearing3_3"]
    if config["development_bearings"] != bearings:
        raise ValueError("development scope changed")
    if config["protected_bearings"] != ["Bearing3_4", "Bearing3_5"]:
        raise ValueError("protected scope changed")
    output = (ROOT / config["output_directory"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnosis: {output}")

    cache = (ROOT / config["input"]["cache_directory"]).resolve()
    scaled: dict[str, np.ndarray] = {}
    for entry in config["input"]["entries"]:
        bearing = entry["bearing_id"]
        if bearing not in bearings:
            raise ValueError("cache entry outside development scope")
        path = cache / f"{bearing}.npz"
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"cache hash mismatch: {bearing}")
        with np.load(path, allow_pickle=False) as payload:
            features = torch.from_numpy(
                np.asarray(payload["features"], dtype=np.float32)
            )
        scaler = StableCausalFeatureScaler.fit(
            features,
            calibration_steps=config["input"]["calibration_steps"],
            mode=config["input"]["scaling_mode"],
        )
        scaled[bearing] = scaler.transform(features).numpy()

    magnitude = config["diagnostics"]["feature_lifetime_spearman"][
        "robust_magnitude"
    ]
    full_correlations = {
        bearing: feature_lifetime_spearman(scaled[bearing]) for bearing in bearings
    }
    feature_rows = []
    for index, name in enumerate(FEATURE_NAMES):
        values = np.asarray(
            [full_correlations[bearing][index] for bearing in bearings]
        )
        strong = np.abs(values) >= magnitude
        strong_signs = np.sign(values[strong])
        conflict = bool(
            strong_signs.size >= 2
            and np.any(strong_signs > 0)
            and np.any(strong_signs < 0)
        )
        feature_rows.append(
            {
                "feature": name,
                "rho_Bearing3_1": float(values[0]),
                "rho_Bearing3_2": float(values[1]),
                "rho_Bearing3_3": float(values[2]),
                "strong_in_all_three": bool(strong.all()),
                "same_sign_all_three": bool(
                    np.all(values > 0.0) or np.all(values < 0.0)
                ),
                "robust_direction_conflict": conflict,
            }
        )

    phase_rows = []
    split = config["diagnostics"]["phase_split"]["early_end_fraction"]
    for bearing in bearings:
        values = scaled[bearing]
        boundary = int(np.floor(values.shape[0] * split))
        for phase, segment in (("early_0_60", values[:boundary]), ("late_60_100", values[boundary:])):
            correlations = feature_lifetime_spearman(segment)
            phase_rows.append(
                {
                    "bearing_id": bearing,
                    "phase": phase,
                    "steps": segment.shape[0],
                    "median_absolute_feature_rho": float(np.median(np.abs(correlations))),
                    "strong_feature_count": int(np.count_nonzero(np.abs(correlations) >= magnitude)),
                }
            )

    probe = config["diagnostics"]["bearing_identity_probe"]
    sampled_values = []
    sampled_labels = []
    sampled_blocks = []
    for label, bearing in enumerate(bearings):
        indices = evenly_spaced_indices(
            scaled[bearing].shape[0], probe["equal_samples_per_bearing"]
        )
        life = indices / float(scaled[bearing].shape[0] - 1)
        blocks = np.minimum(
            (life * probe["contiguous_lifetime_folds"]).astype(int),
            probe["contiguous_lifetime_folds"] - 1,
        )
        sampled_values.append(scaled[bearing][indices])
        sampled_labels.append(np.full(indices.size, label, dtype=int))
        sampled_blocks.append(blocks)
    identity_x = np.concatenate(sampled_values)
    identity_y = np.concatenate(sampled_labels)
    identity_blocks = np.concatenate(sampled_blocks)
    identity_rows = []
    predictions = np.empty_like(identity_y)
    for block in range(probe["contiguous_lifetime_folds"]):
        test_mask = identity_blocks == block
        predictions[test_mask] = nearest_centroid_predict(
            identity_x[~test_mask], identity_y[~test_mask], identity_x[test_mask]
        )
        identity_rows.append(
            {
                "held_lifetime_block": block,
                "accuracy": float(np.mean(predictions[test_mask] == identity_y[test_mask])),
                "test_samples": int(np.count_nonzero(test_mask)),
            }
        )
    identity_accuracy = float(np.mean(predictions == identity_y))

    ridge_config = config["diagnostics"]["rul_mapping_probe"]
    ridge_rows = []
    for test_bearing in bearings:
        train_x = []
        train_y = []
        for bearing in bearings:
            if bearing == test_bearing:
                continue
            indices = evenly_spaced_indices(
                scaled[bearing].shape[0], ridge_config["equal_samples_per_bearing"]
            )
            train_x.append(scaled[bearing][indices])
            train_y.append(indices / float(scaled[bearing].shape[0] - 1))
        test_indices = evenly_spaced_indices(
            scaled[test_bearing].shape[0], ridge_config["equal_samples_per_bearing"]
        )
        test_lifetime = test_indices / float(scaled[test_bearing].shape[0] - 1)
        prediction = ridge_fit_predict(
            np.concatenate(train_x),
            np.concatenate(train_y),
            scaled[test_bearing][test_indices],
            alpha=ridge_config["alpha"],
        )
        rho = float(spearmanr(test_lifetime, prediction).statistic)
        ridge_rows.append(
            {
                "test_bearing": test_bearing,
                "spearman_lifetime": rho,
                "mae_lifetime": float(np.mean(np.abs(prediction - test_lifetime))),
            }
        )

    temporal_rows = []
    for bearing in bearings:
        for lag in config["diagnostics"]["temporal_scale"]["lags"]:
            temporal_rows.append(
                {
                    "bearing_id": bearing,
                    "lag_steps": lag,
                    "lifetime_fraction": lag / float(scaled[bearing].shape[0] - 1),
                    "median_feature_autocorrelation": median_feature_autocorrelation(
                        scaled[bearing], lag
                    ),
                }
            )

    output.mkdir(parents=True)
    write_csv(output / "feature_direction_correlations.csv", feature_rows)
    write_csv(output / "phase_feature_summary.csv", phase_rows)
    write_csv(output / "bearing_identity_probe.csv", identity_rows)
    write_csv(output / "ridge_lobo_probe.csv", ridge_rows)
    write_csv(output / "temporal_scale.csv", temporal_rows)
    thresholds = config["interpretation_thresholds"]
    report = {
        "diagnosis_id": config["diagnosis_id"],
        "status": "completed",
        "bearing_identity_accuracy": identity_accuracy,
        "strong_bearing_identity": identity_accuracy
        >= thresholds["strong_bearing_identity_accuracy"],
        "robust_direction_conflict_feature_count": int(
            sum(bool(row["robust_direction_conflict"]) for row in feature_rows)
        ),
        "strong_same_sign_all_three_feature_count": int(
            sum(
                bool(row["strong_in_all_three"] and row["same_sign_all_three"])
                for row in feature_rows
            )
        ),
        "ridge_lobo": ridge_rows,
        "unstable_cross_bearing_rul_mapping": any(
            float(row["spearman_lifetime"])
            < thresholds["unstable_rul_mapping_if_any_lobo_spearman_below"]
            for row in ridge_rows
        ),
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "feature_lstm_training_performed": False,
        "input_cache_hashes_verified": True,
    }
    (output / "diagnosis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
