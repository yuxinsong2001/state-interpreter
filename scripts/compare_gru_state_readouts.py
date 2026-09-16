"""Compare causal state readouts from frozen predictive-GRU checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from itertools import combinations
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
repository_src = repository_root / "src"
scripts_dir = repository_root / "scripts"
for path in (repository_src, scripts_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from state_interpreter import PredictiveGRUStateInterpreter
from run_gru_interpreter_exploratory import (
    center_sequence,
    gru_level,
    load_latents,
    sha256,
    to_latent,
    trajectory_metrics,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def smooth_after(
    raw_values: np.ndarray,
    *,
    output_start: int,
    temporal_window: int,
) -> np.ndarray:
    result = np.full(len(raw_values), np.nan, dtype=float)
    history: deque[float] = deque(maxlen=temporal_window)
    for index in range(output_start, len(raw_values)):
        value = float(raw_values[index])
        if not math.isfinite(value):
            raise ValueError(f"non-finite readout value at index {index}")
        history.append(value)
        result[index] = float(np.mean(history))
    return result


def predicted_z_readouts(
    standardized_sequence: torch.Tensor,
    predictions_from_current: torch.Tensor,
    *,
    calibration_steps: int,
    temporal_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(standardized_sequence) <= calibration_steps + 1:
        raise ValueError("sequence is too short for aligned predictive readouts")
    aligned_prediction = torch.full_like(standardized_sequence, torch.nan)
    aligned_prediction[1:] = predictions_from_current[:-1]
    reference = aligned_prediction[1 : calibration_steps + 1].mean(dim=0)
    predicted_distance_raw = torch.full(
        (len(standardized_sequence),), torch.nan, dtype=standardized_sequence.dtype
    )
    residual_raw = torch.full_like(predicted_distance_raw, torch.nan)
    output_start = calibration_steps + 1
    predicted_distance_raw[output_start:] = torch.linalg.vector_norm(
        aligned_prediction[output_start:] - reference, dim=1
    )
    residual_raw[output_start:] = torch.linalg.vector_norm(
        standardized_sequence[output_start:] - aligned_prediction[output_start:],
        dim=1,
    )
    predicted_distance = smooth_after(
        predicted_distance_raw.cpu().numpy(),
        output_start=output_start,
        temporal_window=temporal_window,
    )
    residual = smooth_after(
        residual_raw.cpu().numpy(),
        output_start=output_start,
        temporal_window=temporal_window,
    )
    return (
        predicted_distance,
        residual,
        predicted_distance_raw.cpu().numpy(),
        residual_raw.cpu().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (repository_root / config["input_latents"]).resolve()
    multiseed_dir = (repository_root / config["multiseed_result_dir"]).resolve()
    output_dir = (repository_root / config["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()

    seeds = [int(seed) for seed in config["random_seeds"]]
    calibration_steps = int(config["calibration_steps"])
    temporal_window = int(config["temporal_window"])
    common_start = int(config["common_score_start_step"])
    if common_start < calibration_steps + temporal_window:
        raise ValueError("common score start must include a full post-calibration window")

    by_bearing, latent_columns = load_latents(input_path)
    raw_latents = {
        bearing: to_latent(rows, latent_columns)
        for bearing, rows in by_bearing.items()
    }
    split_by_bearing = {
        bearing: rows[0]["split"] for bearing, rows in by_bearing.items()
    }
    archived_hidden = {
        (int(row["seed"]), row["bearing_id"], int(row["step_id"])): row["gru_level"]
        for row in load_csv(multiseed_dir / "per_step_gru_levels.csv")
    }

    per_step_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    levels: dict[tuple[str, int, str], np.ndarray] = {}
    hidden_reproduction_max_difference = 0.0
    checkpoint_hashes: dict[str, str] = {}

    for seed in seeds:
        checkpoint_path = multiseed_dir / f"seed_{seed}_checkpoint.pt"
        checkpoint_hashes[str(seed)] = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint["random_seed"]) != seed:
            raise ValueError(f"checkpoint seed mismatch for {seed}")
        if checkpoint["input_sha256"] != sha256(input_path):
            raise ValueError(f"input hash mismatch for seed {seed}")
        model = PredictiveGRUStateInterpreter(
            embedding_dim=int(checkpoint["embedding_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        feature_mean = checkpoint["feature_mean"]
        feature_std = checkpoint["feature_std"]

        with torch.no_grad():
            for bearing in sorted(by_bearing):
                centered = center_sequence(raw_latents[bearing], calibration_steps)
                standardized = (centered - feature_mean) / feature_std
                output = model(standardized[None, :, :])
                hidden_level, hidden_raw = gru_level(
                    output.hidden_sequence[0], calibration_steps, temporal_window
                )
                (
                    predicted_level,
                    residual_level,
                    predicted_raw,
                    residual_raw,
                ) = predicted_z_readouts(
                    standardized,
                    output.next_embedding_prediction[0],
                    calibration_steps=calibration_steps,
                    temporal_window=temporal_window,
                )
                readouts = {
                    "hidden_distance": hidden_level,
                    "predicted_z_distance": predicted_level,
                    "prediction_residual": residual_level,
                }
                raw_readouts = {
                    "hidden_distance": hidden_raw,
                    "predicted_z_distance": predicted_raw,
                    "prediction_residual": residual_raw,
                }
                normalized_lifetime = np.asarray(
                    [float(row["normalized_lifetime"]) for row in by_bearing[bearing]],
                    dtype=float,
                )
                for method, values in readouts.items():
                    score_mask = np.arange(len(values)) >= common_start
                    score_mask &= np.isfinite(values)
                    metrics = trajectory_metrics(
                        normalized_lifetime, values, score_mask
                    )
                    metric_rows.append(
                        {
                            "seed": seed,
                            "bearing_id": bearing,
                            "split": split_by_bearing[bearing],
                            "readout": method,
                            **metrics,
                        }
                    )
                    levels[(method, seed, bearing)] = values
                for index, row in enumerate(by_bearing[bearing]):
                    archived = archived_hidden[(seed, bearing, int(row["step_id"]))]
                    if archived != "" and math.isfinite(hidden_level[index]):
                        hidden_reproduction_max_difference = max(
                            hidden_reproduction_max_difference,
                            abs(float(archived) - float(hidden_level[index])),
                        )
                    per_step_rows.append(
                        {
                            "seed": seed,
                            "bearing_id": bearing,
                            "split": split_by_bearing[bearing],
                            "step_id": int(row["step_id"]),
                            "normalized_lifetime": normalized_lifetime[index],
                            "included_in_common_score": index >= common_start,
                            "hidden_distance_level": (
                                "" if math.isnan(hidden_level[index]) else hidden_level[index]
                            ),
                            "predicted_z_distance_level": (
                                "" if math.isnan(predicted_level[index]) else predicted_level[index]
                            ),
                            "prediction_residual_level": (
                                "" if math.isnan(residual_level[index]) else residual_level[index]
                            ),
                            "hidden_distance_raw": hidden_raw[index],
                            "predicted_z_distance_raw": (
                                "" if math.isnan(predicted_raw[index]) else predicted_raw[index]
                            ),
                            "prediction_residual_raw": (
                                "" if math.isnan(residual_raw[index]) else residual_raw[index]
                            ),
                        }
                    )

    consistency_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for readout in config["readouts"]:
        for bearing in sorted(by_bearing):
            selected_metrics = [
                row
                for row in metric_rows
                if row["readout"] == readout and row["bearing_id"] == bearing
            ]
            rhos = np.asarray([float(row["spearman_rho"]) for row in selected_metrics])
            pairwise_pearson: list[float] = []
            pairwise_spearman: list[float] = []
            for left_seed, right_seed in combinations(seeds, 2):
                left = levels[(readout, left_seed, bearing)]
                right = levels[(readout, right_seed, bearing)]
                mask = (np.arange(len(left)) >= common_start) & np.isfinite(left) & np.isfinite(right)
                pearson_value = float(pearsonr(left[mask], right[mask]).statistic)
                spearman_value = float(spearmanr(left[mask], right[mask]).statistic)
                pairwise_pearson.append(pearson_value)
                pairwise_spearman.append(spearman_value)
                consistency_rows.append(
                    {
                        "readout": readout,
                        "bearing_id": bearing,
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "pearson_r": pearson_value,
                        "spearman_rho": spearman_value,
                        "samples": int(mask.sum()),
                    }
                )
            summary_rows.append(
                {
                    "readout": readout,
                    "bearing_id": bearing,
                    "split": split_by_bearing[bearing],
                    "rho_mean": float(rhos.mean()),
                    "rho_sample_std": float(rhos.std(ddof=1)),
                    "rho_min": float(rhos.min()),
                    "rho_max": float(rhos.max()),
                    "collapse_count": int(
                        sum(bool(row["collapsed"]) for row in selected_metrics)
                    ),
                    "mean_pairwise_pearson": float(np.mean(pairwise_pearson)),
                    "minimum_pairwise_pearson": float(np.min(pairwise_pearson)),
                    "mean_pairwise_spearman": float(np.mean(pairwise_spearman)),
                    "minimum_pairwise_spearman": float(np.min(pairwise_spearman)),
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for readout in config["readouts"]:
        selected = [row for row in summary_rows if row["readout"] == readout]
        holdout_rows = [
            row for row in selected if row["bearing_id"] in {"Bearing1_4", "Bearing1_5"}
        ]
        aggregate_rows.append(
            {
                "readout": readout,
                "mean_rho_across_bearings_descriptive": float(
                    np.mean([float(row["rho_mean"]) for row in selected])
                ),
                "mean_seed_std_across_bearings": float(
                    np.mean([float(row["rho_sample_std"]) for row in selected])
                ),
                "mean_seed_std_bearing1_4_and_1_5": float(
                    np.mean([float(row["rho_sample_std"]) for row in holdout_rows])
                ),
                "minimum_rho_over_all_seed_bearing_pairs": float(
                    min(
                        float(row["spearman_rho"])
                        for row in metric_rows
                        if row["readout"] == readout
                    )
                ),
                "total_collapse_count": int(
                    sum(bool(row["collapsed"]) for row in metric_rows if row["readout"] == readout)
                ),
            }
        )

    for filename, rows in (
        ("per_step_readouts.csv", per_step_rows),
        ("per_seed_bearing_metrics.csv", metric_rows),
        ("trajectory_consistency.csv", consistency_rows),
        ("summary_by_readout_bearing.csv", summary_rows),
        ("readout_aggregate.csv", aggregate_rows),
    ):
        write_csv(output_dir / filename, rows)

    report = {
        "status": "completed",
        "verification_status": "fixed_checkpoint_ablation",
        "experiment_id": config["experiment_id"],
        "input_sha256": sha256(input_path),
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": checkpoint_hashes,
        "random_seeds": seeds,
        "common_score_start_step": common_start,
        "hidden_readout_reproduction_max_absolute_difference": hidden_reproduction_max_difference,
        "bearing_summaries": summary_rows,
        "readout_aggregates": aggregate_rows,
        "warnings": [
            "All bearings and checkpoints were inspected previously; this is not a blind test.",
            "The common scoring interval starts at step 25 and differs from the earlier step-15 report.",
            "Prediction residual is a change/anomaly candidate and is not assumed to be monotonic.",
            "Three seeds are repeated models on the same bearings, not independent equipment samples.",
            "Normalized lifetime is used only for descriptive evaluation.",
        ],
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "per_step_readouts.csv",
            "per_seed_bearing_metrics.csv",
            "trajectory_consistency.csv",
            "summary_by_readout_bearing.csv",
            "readout_aggregate.csv",
            "experiment_report.json",
        ],
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
