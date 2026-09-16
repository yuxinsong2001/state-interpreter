"""Repeat the fixed predictive-GRU experiment across random seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from copy import deepcopy
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
    load_centered_hmm_metrics,
    load_latents,
    prediction_mse,
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


def train_one_seed(
    *,
    seed: int,
    config: dict[str, object],
    train_sequences: list[torch.Tensor],
    validation_sequence: torch.Tensor,
    embedding_dim: int,
) -> tuple[
    PredictiveGRUStateInterpreter,
    list[dict[str, object]],
    int,
    float,
]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PredictiveGRUStateInterpreter(
        embedding_dim=embedding_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = prediction_mse(model, train_sequences)
        train_loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(prediction_mse(model, [validation_sequence]))
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_next_step_mse": float(train_loss.detach()),
                "validation_next_step_mse": validation_loss,
                "gradient_norm_before_clip": gradient_norm,
            }
        )
        if validation_loss < best_validation - float(config["minimum_improvement"]):
            best_validation = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"seed {seed} did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, best_validation


def compare_checkpoint_parameters(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> tuple[float, int]:
    if set(left) != set(right):
        raise ValueError("checkpoint parameter keys differ")
    maximum = 0.0
    mismatch_count = 0
    for name in left:
        difference = (left[name] - right[name]).abs()
        maximum = max(maximum, float(difference.max()))
        mismatch_count += int(torch.count_nonzero(difference))
    return maximum, mismatch_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (repository_root / config["input_latents"]).resolve()
    hmm_metrics_path = (repository_root / config["centered_hmm_metrics"]).resolve()
    single_seed_dir = (repository_root / config["single_seed_result"]).resolve()
    output_dir = (repository_root / config["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()

    seeds = [int(seed) for seed in config["random_seeds"]]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three unique random seeds are required")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    by_bearing, latent_columns = load_latents(input_path)
    raw_latents = {
        bearing: to_latent(rows, latent_columns)
        for bearing, rows in by_bearing.items()
    }
    split_by_bearing = {
        bearing: rows[0]["split"] for bearing, rows in by_bearing.items()
    }
    train_bearings = sorted(
        bearing
        for bearing, split in split_by_bearing.items()
        if split == config["training_split"]
    )
    validation_bearing = str(config["validation_bearing"])
    holdout_bearing = str(config["old_holdout_bearing"])
    calibration_steps = int(config["calibration_steps"])
    centered = {
        bearing: center_sequence(sequence, calibration_steps)
        for bearing, sequence in raw_latents.items()
    }
    train_values = torch.cat([centered[bearing] for bearing in train_bearings])
    feature_mean = train_values.mean(dim=0)
    feature_std = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = {
        bearing: (sequence - feature_mean) / feature_std
        for bearing, sequence in centered.items()
    }
    train_sequences = [standardized[bearing] for bearing in train_bearings]
    hmm_metrics = load_centered_hmm_metrics(hmm_metrics_path)
    original_comparison = {
        row["bearing_id"]: row
        for row in load_csv(single_seed_dir / "method_comparison.csv")
    }
    original_checkpoint = torch.load(
        single_seed_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )

    history_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    seed_summary_rows: list[dict[str, object]] = []
    levels_by_seed_bearing: dict[tuple[int, str], np.ndarray] = {}
    reproduction: dict[str, object] | None = None

    for seed in seeds:
        model, history, best_epoch, best_validation = train_one_seed(
            seed=seed,
            config=config,
            train_sequences=train_sequences,
            validation_sequence=standardized[validation_bearing],
            embedding_dim=len(latent_columns),
        )
        history_rows.extend(history)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "embedding_dim": len(latent_columns),
            "hidden_dim": int(config["hidden_dim"]),
            "num_layers": int(config["num_layers"]),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "best_epoch": best_epoch,
            "best_validation_next_step_mse": best_validation,
            "random_seed": seed,
            "input_sha256": sha256(input_path),
            "config_sha256": sha256(config_path),
        }
        torch.save(checkpoint, output_dir / f"seed_{seed}_checkpoint.pt")
        seed_rhos: list[float] = []
        seed_collapses = 0
        metric_by_bearing: dict[str, dict[str, object]] = {}
        with torch.no_grad():
            for bearing in sorted(by_bearing):
                sequence = standardized[bearing]
                output = model(sequence[None, :, :])
                hidden = output.hidden_sequence[0]
                level, hidden_distance = gru_level(
                    hidden,
                    calibration_steps,
                    int(config["temporal_window"]),
                )
                normalized_lifetime = np.asarray(
                    [float(row["normalized_lifetime"]) for row in by_bearing[bearing]],
                    dtype=float,
                )
                score_mask = np.arange(len(level)) >= int(
                    config["common_score_start_step"]
                )
                score_mask &= np.isfinite(level)
                metrics = trajectory_metrics(normalized_lifetime, level, score_mask)
                difference = (
                    output.next_embedding_prediction[0, :-1] - sequence[1:]
                )
                next_step_mse = float(difference.square().mean())
                row = {
                    "seed": seed,
                    "bearing_id": bearing,
                    "split": split_by_bearing[bearing],
                    **metrics,
                    "next_step_mse": next_step_mse,
                    "best_epoch": best_epoch,
                    "best_validation_next_step_mse": best_validation,
                }
                metric_rows.append(row)
                metric_by_bearing[bearing] = row
                seed_rhos.append(float(metrics["spearman_rho"]))
                seed_collapses += int(bool(metrics["collapsed"]))
                levels_by_seed_bearing[(seed, bearing)] = level
                for index, value in enumerate(level):
                    level_rows.append(
                        {
                            "seed": seed,
                            "bearing_id": bearing,
                            "split": split_by_bearing[bearing],
                            "step_id": int(by_bearing[bearing][index]["step_id"]),
                            "normalized_lifetime": normalized_lifetime[index],
                            "gru_level": "" if math.isnan(value) else value,
                            "gru_hidden_distance": hidden_distance[index],
                            "included_in_common_score": bool(score_mask[index]),
                        }
                    )
        seed_summary_rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_next_step_mse": best_validation,
                "mean_bearing_rho_descriptive": float(np.mean(seed_rhos)),
                "minimum_bearing_rho": float(np.min(seed_rhos)),
                "maximum_bearing_rho": float(np.max(seed_rhos)),
                "collapse_count": seed_collapses,
            }
        )
        if seed == seeds[0]:
            parameter_max_diff, parameter_mismatches = compare_checkpoint_parameters(
                model.state_dict(), original_checkpoint["model_state_dict"]
            )
            rho_differences = [
                abs(
                    float(metric_by_bearing[bearing]["spearman_rho"])
                    - float(original_comparison[bearing]["predictive_gru_rho"])
                )
                for bearing in sorted(by_bearing)
            ]
            mse_differences = [
                abs(
                    float(metric_by_bearing[bearing]["next_step_mse"])
                    - float(original_comparison[bearing]["gru_next_step_mse"])
                )
                for bearing in sorted(by_bearing)
            ]
            reproduction = {
                "seed": seed,
                "best_epoch_match": best_epoch == int(original_checkpoint["best_epoch"]),
                "validation_mse_absolute_difference": abs(
                    best_validation
                    - float(original_checkpoint["best_validation_next_step_mse"])
                ),
                "parameter_max_absolute_difference": parameter_max_diff,
                "parameter_nonzero_difference_count": parameter_mismatches,
                "maximum_bearing_rho_absolute_difference": max(rho_differences),
                "maximum_bearing_next_step_mse_absolute_difference": max(mse_differences),
            }
            reproduction["exact_match"] = all(
                [
                    bool(reproduction["best_epoch_match"]),
                    float(reproduction["validation_mse_absolute_difference"]) == 0.0,
                    parameter_max_diff == 0.0,
                    parameter_mismatches == 0,
                    max(rho_differences) == 0.0,
                    max(mse_differences) == 0.0,
                ]
            )

    if reproduction is None:
        raise RuntimeError("the first seed reproduction check was not executed")

    trajectory_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    for bearing in sorted(by_bearing):
        bearing_metric_rows = [row for row in metric_rows if row["bearing_id"] == bearing]
        rhos = np.asarray([float(row["spearman_rho"]) for row in bearing_metric_rows])
        prediction_mses = np.asarray(
            [float(row["next_step_mse"]) for row in bearing_metric_rows]
        )
        pair_pearson: list[float] = []
        pair_spearman: list[float] = []
        for left_seed, right_seed in combinations(seeds, 2):
            left = levels_by_seed_bearing[(left_seed, bearing)]
            right = levels_by_seed_bearing[(right_seed, bearing)]
            mask = np.isfinite(left) & np.isfinite(right)
            pearson_value = float(pearsonr(left[mask], right[mask]).statistic)
            spearman_value = float(spearmanr(left[mask], right[mask]).statistic)
            pair_pearson.append(pearson_value)
            pair_spearman.append(spearman_value)
            trajectory_rows.append(
                {
                    "bearing_id": bearing,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "pearson_r": pearson_value,
                    "spearman_rho": spearman_value,
                    "samples": int(mask.sum()),
                }
            )
        summary = {
            "bearing_id": bearing,
            "split": split_by_bearing[bearing],
            "gru_rho_mean": float(rhos.mean()),
            "gru_rho_sample_std": float(rhos.std(ddof=1)),
            "gru_rho_min": float(rhos.min()),
            "gru_rho_max": float(rhos.max()),
            "gru_next_step_mse_mean": float(prediction_mses.mean()),
            "gru_next_step_mse_sample_std": float(prediction_mses.std(ddof=1)),
            "collapse_count": int(sum(bool(row["collapsed"]) for row in bearing_metric_rows)),
            "mean_pairwise_level_pearson": float(np.mean(pair_pearson)),
            "minimum_pairwise_level_pearson": float(np.min(pair_pearson)),
            "mean_pairwise_level_spearman": float(np.mean(pair_spearman)),
            "minimum_pairwise_level_spearman": float(np.min(pair_spearman)),
        }
        summary_rows.append(summary)
        baseline = original_comparison[bearing]
        method_rows.append(
            {
                "bearing_id": bearing,
                "split": split_by_bearing[bearing],
                "distance_rho": float(baseline["distance_rho"]),
                "centered_hmm_rho": float(hmm_metrics[bearing]["spearman_rho"]),
                "gru_rho_mean": summary["gru_rho_mean"],
                "gru_rho_sample_std": summary["gru_rho_sample_std"],
                "gru_mean_minus_distance": float(summary["gru_rho_mean"])
                - float(baseline["distance_rho"]),
                "gru_mean_minus_centered_hmm": float(summary["gru_rho_mean"])
                - float(hmm_metrics[bearing]["spearman_rho"]),
            }
        )

    for filename, rows in (
        ("training_history.csv", history_rows),
        ("per_seed_bearing_metrics.csv", metric_rows),
        ("per_step_gru_levels.csv", level_rows),
        ("per_seed_summary.csv", seed_summary_rows),
        ("trajectory_consistency.csv", trajectory_rows),
        ("summary_by_bearing.csv", summary_rows),
        ("method_comparison.csv", method_rows),
    ):
        write_csv(output_dir / filename, rows)
    (output_dir / "reproduction_check.json").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "status": "completed",
        "verification_status": (
            "verified_multiseed_exploratory"
            if bool(reproduction["exact_match"])
            else "reproduction_mismatch"
        ),
        "experiment_id": config["experiment_id"],
        "input_sha256": sha256(input_path),
        "config_sha256": sha256(config_path),
        "single_seed_comparison_sha256": sha256(
            single_seed_dir / "method_comparison.csv"
        ),
        "random_seeds": seeds,
        "train_bearings": train_bearings,
        "validation_bearing": validation_bearing,
        "old_holdout_bearing": holdout_bearing,
        "reproduction": reproduction,
        "seed_summaries": seed_summary_rows,
        "bearing_summaries": summary_rows,
        "method_comparisons": method_rows,
        "warnings": [
            "All bearings were inspected previously; this is not a blind test.",
            "Three seeds provide preliminary, not definitive, stability evidence.",
            "Seed runs on the same bearing are not independent equipment samples.",
            "Bearing1_4 selects every checkpoint and is not independent test evidence.",
            "Normalized lifetime is used only for descriptive evaluation.",
        ],
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "seed_*_checkpoint.pt",
            "training_history.csv",
            "per_seed_bearing_metrics.csv",
            "per_step_gru_levels.csv",
            "per_seed_summary.csv",
            "trajectory_consistency.csv",
            "summary_by_bearing.csv",
            "method_comparison.csv",
            "reproduction_check.json",
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
