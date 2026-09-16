"""Train and evaluate a small self-supervised predictive GRU interpreter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
repository_src = repository_root / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import numpy as np
import torch
from scipy.stats import spearmanr

from state_interpreter import (
    PredictiveGRUStateInterpreter,
    RelativeTemporalStateInterpreter,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def load_latents(path: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("latent CSV is empty")
    latent_columns = sorted(
        (name for name in rows[0] if name.startswith("z_")),
        key=lambda name: int(name.split("_")[1]),
    )
    if not latent_columns:
        raise ValueError("latent CSV has no z_* columns")
    by_bearing: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_bearing.setdefault(row["bearing_id"], []).append(row)
    for bearing, bearing_rows in by_bearing.items():
        bearing_rows.sort(key=lambda row: int(row["step_id"]))
        steps = [int(row["step_id"]) for row in bearing_rows]
        if steps != sorted(steps) or len(steps) != len(set(steps)):
            raise ValueError(f"invalid step order for {bearing}")
    return by_bearing, latent_columns


def to_latent(rows: list[dict[str, str]], columns: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=torch.float32,
    )


def center_sequence(sequence: torch.Tensor, calibration_steps: int) -> torch.Tensor:
    if calibration_steps < 1 or calibration_steps > len(sequence):
        raise ValueError("calibration_steps must be within each sequence")
    return sequence - sequence[:calibration_steps].mean(dim=0, keepdim=True)


def prediction_mse(
    model: PredictiveGRUStateInterpreter,
    sequences: list[torch.Tensor],
) -> torch.Tensor:
    squared_error = torch.zeros((), dtype=torch.float32)
    value_count = 0
    for sequence in sequences:
        output = model(sequence[None, :, :])
        difference = output.next_embedding_prediction[:, :-1] - sequence[None, 1:]
        squared_error = squared_error + difference.square().sum()
        value_count += difference.numel()
    return squared_error / value_count


def distance_baseline_level(
    latent: torch.Tensor,
    calibration_steps: int,
    temporal_window: int,
) -> np.ndarray:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=latent.shape[1],
        calibration_steps=calibration_steps,
        temporal_window=temporal_window,
    )
    level = np.full(len(latent), np.nan, dtype=float)
    for index, z in enumerate(latent):
        output = interpreter.update(z)
        if output is not None:
            level[index] = float(output.level)
    return level


def gru_level(
    hidden: torch.Tensor,
    calibration_steps: int,
    temporal_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference = hidden[:calibration_steps].mean(dim=0)
    raw_distance = torch.linalg.vector_norm(hidden - reference, dim=1).cpu().numpy()
    level = np.full(len(hidden), np.nan, dtype=float)
    history: deque[float] = deque(maxlen=temporal_window)
    for index in range(calibration_steps, len(hidden)):
        history.append(float(raw_distance[index]))
        level[index] = float(np.mean(history))
    return level, raw_distance


def trajectory_metrics(
    normalized_lifetime: np.ndarray,
    values: np.ndarray,
    score_mask: np.ndarray,
) -> dict[str, float | int | bool]:
    selected = values[score_mask]
    lifetime = normalized_lifetime[score_mask]
    differences = np.diff(selected)
    return {
        "score_samples": int(len(selected)),
        "spearman_rho": safe_spearman(lifetime, selected),
        "first_last_delta": float(selected[-1] - selected[0]),
        "mean_absolute_step": float(np.mean(np.abs(differences))),
        "step_std": float(np.std(differences)),
        "backward_step_fraction": float(np.mean(differences < -1e-9)),
        "collapsed": bool(np.ptp(selected) == 0),
    }


def load_centered_hmm_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["bearing_id"]: row
        for row in rows
        if row["representation"] == "bearing_centered_z"
    }
    if not selected:
        raise ValueError("centered HMM metric rows are missing")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (repository_root / config["input_latents"]).resolve()
    hmm_metrics_path = (repository_root / config["centered_hmm_metrics"]).resolve()
    output_dir = (repository_root / config["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()

    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    if set(train_bearings) & {validation_bearing, holdout_bearing}:
        raise ValueError("train, validation and holdout bearings must be disjoint")
    if validation_bearing not in raw_latents or holdout_bearing not in raw_latents:
        raise ValueError("validation or holdout bearing is missing")

    calibration_steps = int(config["calibration_steps"])
    centered_latents = {
        bearing: center_sequence(sequence, calibration_steps)
        for bearing, sequence in raw_latents.items()
    }
    train_values = torch.cat([centered_latents[bearing] for bearing in train_bearings])
    feature_mean = train_values.mean(dim=0)
    feature_std = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = {
        bearing: (sequence - feature_mean) / feature_std
        for bearing, sequence in centered_latents.items()
    }

    model = PredictiveGRUStateInterpreter(
        embedding_dim=len(latent_columns),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_sequences = [standardized[bearing] for bearing in train_bearings]
    validation_sequences = [standardized[validation_bearing]]
    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

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
            validation_loss = float(prediction_mse(model, validation_sequences))
        train_value = float(train_loss.detach())
        history_rows.append(
            {
                "epoch": epoch,
                "train_next_step_mse": train_value,
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
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_state,
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
        },
        checkpoint_path,
    )
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)

    hmm_metrics = load_centered_hmm_metrics(hmm_metrics_path)
    per_step_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    prediction_mse_by_bearing: dict[str, float] = {}

    with torch.no_grad():
        for bearing in sorted(by_bearing):
            rows = by_bearing[bearing]
            normalized_lifetime = np.asarray(
                [float(row["normalized_lifetime"]) for row in rows], dtype=float
            )
            output = model(standardized[bearing][None, :, :])
            hidden = output.hidden_sequence[0]
            prediction_difference = (
                output.next_embedding_prediction[0, :-1] - standardized[bearing][1:]
            )
            per_prediction_mse = prediction_difference.square().mean(dim=1).cpu().numpy()
            prediction_mse_by_bearing[bearing] = float(
                prediction_difference.square().mean()
            )
            baseline = distance_baseline_level(
                raw_latents[bearing],
                calibration_steps,
                int(config["temporal_window"]),
            )
            learned_level, hidden_distance = gru_level(
                hidden,
                calibration_steps,
                int(config["temporal_window"]),
            )
            common_mask = np.arange(len(rows)) >= int(config["common_score_start_step"])
            common_mask &= np.isfinite(baseline) & np.isfinite(learned_level)
            baseline_metrics = trajectory_metrics(
                normalized_lifetime, baseline, common_mask
            )
            gru_metrics = trajectory_metrics(
                normalized_lifetime, learned_level, common_mask
            )
            hmm_row = hmm_metrics[bearing]
            hmm_rho = float(hmm_row["spearman_rho"])
            metric_rows.extend(
                [
                    {
                        "bearing_id": bearing,
                        "split": split_by_bearing[bearing],
                        "method": "distance_baseline",
                        **baseline_metrics,
                        "next_step_mse": "",
                    },
                    {
                        "bearing_id": bearing,
                        "split": split_by_bearing[bearing],
                        "method": "centered_hmm",
                        "score_samples": int(hmm_row["score_samples"]),
                        "spearman_rho": hmm_rho,
                        "first_last_delta": float(hmm_row["first_last_delta"]),
                        "mean_absolute_step": float(hmm_row["mean_absolute_step"]),
                        "step_std": float(hmm_row["step_std"]),
                        "backward_step_fraction": float(hmm_row["backward_step_fraction"]),
                        "collapsed": hmm_row["collapsed_to_one_stage"] == "True",
                        "next_step_mse": "",
                    },
                    {
                        "bearing_id": bearing,
                        "split": split_by_bearing[bearing],
                        "method": "predictive_gru",
                        **gru_metrics,
                        "next_step_mse": prediction_mse_by_bearing[bearing],
                    },
                ]
            )
            comparison_rows.append(
                {
                    "bearing_id": bearing,
                    "split": split_by_bearing[bearing],
                    "distance_rho": baseline_metrics["spearman_rho"],
                    "centered_hmm_rho": hmm_rho,
                    "predictive_gru_rho": gru_metrics["spearman_rho"],
                    "gru_minus_distance": float(gru_metrics["spearman_rho"])
                    - float(baseline_metrics["spearman_rho"]),
                    "gru_minus_centered_hmm": float(gru_metrics["spearman_rho"])
                    - hmm_rho,
                    "gru_collapsed": gru_metrics["collapsed"],
                    "gru_next_step_mse": prediction_mse_by_bearing[bearing],
                }
            )
            for index, row in enumerate(rows):
                step_row: dict[str, object] = {
                    "bearing_id": bearing,
                    "split": split_by_bearing[bearing],
                    "step_id": int(row["step_id"]),
                    "normalized_lifetime": normalized_lifetime[index],
                    "distance_level": "" if math.isnan(baseline[index]) else baseline[index],
                    "gru_level": "" if math.isnan(learned_level[index]) else learned_level[index],
                    "gru_hidden_distance": hidden_distance[index],
                    "gru_next_step_mse": (
                        per_prediction_mse[index]
                        if index < len(per_prediction_mse)
                        else ""
                    ),
                    "included_in_common_score": bool(common_mask[index]),
                }
                step_row.update(
                    {
                        f"gru_hidden_{dimension}": float(hidden[index, dimension])
                        for dimension in range(hidden.shape[1])
                    }
                )
                per_step_rows.append(step_row)

    for filename, rows in (
        ("per_step_states.csv", per_step_rows),
        ("bearing_metrics.csv", metric_rows),
        ("method_comparison.csv", comparison_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "status": "completed",
        "verification_status": "exploratory_single_seed",
        "experiment_id": config["experiment_id"],
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "centered_hmm_metrics_sha256": sha256(hmm_metrics_path),
        "train_bearings": train_bearings,
        "validation_bearing": validation_bearing,
        "old_holdout_bearing": holdout_bearing,
        "model": {
            "type": "one-layer causal predictive GRU",
            "embedding_dim": len(latent_columns),
            "hidden_dim": int(config["hidden_dim"]),
            "parameter_count": parameter_count,
            "training_target": "next standardized bearing-centered embedding",
            "uses_normalized_lifetime_for_training": False,
        },
        "training": {
            "epochs_completed": len(history_rows),
            "best_epoch": best_epoch,
            "best_validation_next_step_mse": best_validation,
            "final_train_next_step_mse": history_rows[-1]["train_next_step_mse"],
            "stopped_by_patience": len(history_rows) < int(config["max_epochs"]),
            "random_seed": seed,
        },
        "prediction_mse_by_bearing": prediction_mse_by_bearing,
        "comparisons": comparison_rows,
        "warnings": [
            "All bearings were inspected previously; this is not a blind test.",
            "This is a single-seed exploratory result and does not establish training stability.",
            "Bearing1_4 selects the checkpoint and is not independent test evidence.",
            "Normalized lifetime is used only for descriptive evaluation, not training.",
            "A predictive hidden-state distance is not a physical damage measurement.",
        ],
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "best_checkpoint.pt",
            "training_history.csv",
            "per_step_states.csv",
            "bearing_metrics.csv",
            "method_comparison.csv",
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
