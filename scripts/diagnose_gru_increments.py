"""Diagnose bias, direction and magnitude in frozen GRU latent increments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
# Fail on a missing plotting dependency before creating experiment artifacts.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from state_interpreter import PredictiveGRUStateInterpreter
from state_interpreter.forecast_evaluation import aligned_squared_errors
from state_interpreter.increment_diagnostics import aligned_increments, increment_metrics
from run_gru_interpreter_exploratory import center_sequence, load_latents, sha256, to_latent


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def per_dimension_metrics(observed: torch.Tensor, predicted: torch.Tensor) -> list[dict]:
    observed, predicted = observed.double(), predicted.double()
    error = predicted - observed
    rows = []
    for dimension in range(observed.shape[1]):
        truth, estimate, residual = observed[:, dimension], predicted[:, dimension], error[:, dimension]
        bias = float(residual.mean())
        raw_mse = float(residual.square().mean())
        centered_mse = float((residual - residual.mean()).square().mean())
        true_rms = float(truth.square().mean().sqrt())
        predicted_rms = float(estimate.square().mean().sqrt())
        denominator = float(truth.square().sum())
        covariance_denominator = float((truth - truth.mean()).square().sum() * (estimate - estimate.mean()).square().sum())
        rows.append({
            "dimension": dimension,
            "true_mean": float(truth.mean()),
            "predicted_mean": float(estimate.mean()),
            "error_bias": bias,
            "raw_mse": raw_mse,
            "bias_mse": bias * bias,
            "bias_fraction": bias * bias / raw_mse if raw_mse > 0 else None,
            "oracle_bias_corrected_mse": centered_mse,
            "true_rms": true_rms,
            "predicted_rms": predicted_rms,
            "magnitude_ratio": predicted_rms / true_rms if true_rms > 0 else None,
            "through_origin_gain": float((truth * estimate).sum()) / denominator if denominator > 0 else None,
            "pearson": float(((truth-truth.mean())*(estimate-estimate.mean())).sum()) / np.sqrt(covariance_denominator) if covariance_denominator > 0 else None,
            "same_sign_fraction": float(((truth * estimate) > 0).double().mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["forecast_horizon"] != 1:
        raise ValueError("only the predeclared one-step horizon is implemented")
    output = (ROOT / cfg["output_directory"]).resolve()
    if not output.is_relative_to(ROOT / "results") or output.exists():
        raise ValueError("output must be a new directory inside results")

    started = time.monotonic()
    checkpoint_dir = ROOT / cfg["checkpoint_directory"]
    persistence_dir = ROOT / cfg["persistence_reference_directory"]
    input_path = ROOT / cfg["input_latents"]
    training_report_path = checkpoint_dir / "experiment_report.json"
    persistence_report_path = persistence_dir / "experiment_report.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    persistence_report = json.loads(persistence_report_path.read_text(encoding="utf-8"))
    if not str(training_report["status"]).startswith("completed") or not str(persistence_report["status"]).startswith("completed"):
        raise ValueError("source experiments are not complete")
    if persistence_report["config"]["primary_target_start_step"] != cfg["primary_target_start_step"]:
        raise ValueError("target start differs from persistence benchmark")
    if persistence_report["config"]["arms"] != cfg["arms"] or persistence_report["config"]["random_seeds"] != cfg["random_seeds"]:
        raise ValueError("arm or seed inventory differs from persistence benchmark")

    training_config_path = ROOT / "configs/xjtu_gru_temporal_ranking_v1.json"
    if sha256(training_config_path) != training_report["config_sha256"]:
        raise ValueError("archived training configuration changed")
    if cfg["calibration_steps"] != training_report["config"]["calibration_steps"]:
        raise ValueError("calibration differs from archived training")
    if sha256(input_path) != training_report["input_sha256"]:
        raise ValueError("input hash differs from archived training")

    checkpoints = {
        (arm, seed): checkpoint_dir / f"{arm}_seed_{seed}_checkpoint.pt"
        for arm in cfg["arms"] for seed in cfg["random_seeds"]
    }
    for path in checkpoints.values():
        if sha256(path) != training_report["checkpoint_sha256"][path.name]:
            raise ValueError(f"checkpoint hash changed: {path.name}")
    protected = [
        config_path, training_config_path, input_path, training_report_path,
        checkpoint_dir / "per_seed_bearing_metrics.csv", persistence_report_path,
        persistence_dir / "per_seed_bearing_metrics.csv", *checkpoints.values(),
    ]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}

    persistence_rows = read_csv(persistence_dir / "per_seed_bearing_metrics.csv")
    persistence_primary = {
        (row["arm"], int(row["seed"]), row["bearing_id"]): row
        for row in persistence_rows if row["scope"] == "primary"
    }
    by_bearing, columns = load_latents(input_path)
    if columns != [f"z_{index}" for index in range(8)]:
        raise ValueError("expected exactly eight latent columns")
    if sorted(by_bearing) != sorted(cfg["bearings"]):
        raise ValueError("unexpected bearing inventory")
    for bearing, rows in by_bearing.items():
        if [int(row["step_id"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"noncontiguous step IDs in {bearing}")

    centered = {
        bearing: center_sequence(to_latent(rows, columns), cfg["calibration_steps"])
        for bearing, rows in by_bearing.items()
    }
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metric_rows: list[dict] = []
    dimension_rows: list[dict] = []
    target_rows: list[dict] = []
    integrity_rows: list[dict] = []
    reference_scaler = None

    for (arm, seed), path in checkpoints.items():
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        if checkpoint["arm"] != arm or checkpoint["random_seed"] != seed or checkpoint["embedding_dim"] != len(columns):
            raise ValueError("checkpoint identity/dimension mismatch")
        if checkpoint["config_sha256"] != training_report["config_sha256"]:
            raise ValueError("checkpoint training config mismatch")
        mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
        if mean.shape != (len(columns),) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid saved normalization statistics")
        if reference_scaler is None:
            reference_scaler = (mean.clone(), std.clone())
        if not torch.equal(mean, reference_scaler[0]) or not torch.equal(std, reference_scaler[1]):
            raise ValueError("arms or seeds have different normalization scales")

        model = PredictiveGRUStateInterpreter(
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for bearing in cfg["bearings"]:
            sequence = (centered[bearing] - mean) / std
            with torch.no_grad():
                forecast = model(sequence[None]).next_embedding_prediction[0]
                prefix_length = min(cfg["prefix_check_length"], len(sequence) - 1)
                prefix = model(sequence[:prefix_length][None]).next_embedding_prediction[0]
            prefix_difference = float((prefix - forecast[:prefix_length]).abs().max())
            if prefix_difference > cfg["prefix_tolerance"]:
                raise ValueError("causal prefix consistency failure")

            observed, predicted = aligned_increments(
                sequence, forecast, target_start_step=cfg["primary_target_start_step"]
            )
            metrics = increment_metrics(observed, predicted)
            reference = persistence_primary[(arm, seed, bearing)]
            direct_errors, _ = aligned_squared_errors(sequence, forecast)
            direct_mse = float(direct_errors[cfg["primary_target_start_step"] - 1 :].double().mean())
            archived_difference = abs(direct_mse - float(reference["gru_mse"]))
            diagnostic_difference = abs(metrics["raw_mse"] - direct_mse)
            if archived_difference != 0:
                raise ValueError(f"persistence-benchmark MSE mismatch for {arm}/{seed}/{bearing}")
            if diagnostic_difference > 1e-7:
                raise ValueError("increment arithmetic differs unexpectedly from direct forecast error")

            flags = cfg["descriptive_flags"]
            metric_rows.append({
                "arm": arm, "seed": seed, "bearing_id": bearing,
                "split": by_bearing[bearing][0]["split"],
                "target_start_step": cfg["primary_target_start_step"],
                **{key: value for key, value in metrics.items() if key != "mean_error_bias"},
                **{f"error_bias_z_{index}": value for index, value in enumerate(metrics["mean_error_bias"])},
                "bias_dominant_flag": metrics["bias_fraction_of_raw_mse"] is not None and metrics["bias_fraction_of_raw_mse"] >= flags["bias_dominant_fraction_minimum"],
                "magnitude_mismatch_flag": metrics["magnitude_ratio"] is not None and not flags["magnitude_ratio_lower"] <= metrics["magnitude_ratio"] <= flags["magnitude_ratio_upper"],
                "poor_direction_flag": (metrics["flattened_cosine"] is None or metrics["flattened_cosine"] <= flags["poor_flattened_direction_maximum"] or metrics["positive_dot_fraction"] <= flags["poor_positive_dot_fraction_maximum"]),
            })
            for row in per_dimension_metrics(observed, predicted):
                dimension_rows.append({"arm": arm, "seed": seed, "bearing_id": bearing, **row})
            for offset in range(len(observed)):
                truth = observed[offset].double()
                estimate = predicted[offset].double()
                truth_norm, estimate_norm = float(truth.norm()), float(estimate.norm())
                dot = float((truth * estimate).sum())
                target_rows.append({
                    "arm": arm, "seed": seed, "bearing_id": bearing,
                    "origin_step": cfg["primary_target_start_step"] - 1 + offset,
                    "target_step": cfg["primary_target_start_step"] + offset,
                    "true_increment_norm": truth_norm,
                    "predicted_increment_norm": estimate_norm,
                    "dot": dot,
                    "cosine": dot / (truth_norm * estimate_norm) if truth_norm > 0 and estimate_norm > 0 else None,
                    "positive_dot": dot > 0,
                    "raw_step_mse": float((estimate - truth).square().mean()),
                    "persistence_step_mse": float(truth.square().mean()),
                })
            integrity_rows.append({
                "arm": arm, "seed": seed, "bearing_id": bearing,
                "archived_mse_absolute_difference": archived_difference,
                "increment_vs_direct_mse_absolute_difference": diagnostic_difference,
                "prefix_max_absolute_difference": prefix_difference,
            })
        print(f"Diagnosed frozen {arm} seed={seed}", flush=True)

    summaries: list[dict] = []
    summary_fields = [
        "raw_mse_ratio", "bias_fraction_of_raw_mse", "oracle_bias_corrected_mse_ratio",
        "magnitude_ratio", "flattened_cosine", "through_origin_gain",
        "positive_dot_fraction", "mean_step_cosine",
    ]
    for arm in cfg["arms"]:
        for bearing in cfg["bearings"]:
            selected = [row for row in metric_rows if row["arm"] == arm and row["bearing_id"] == bearing]
            summary = {
                "arm": arm, "bearing_id": bearing, "split": selected[0]["split"],
                "seed_count": len(selected), "target_count_per_seed": selected[0]["target_count"],
            }
            for field in summary_fields:
                values = [row[field] for row in selected if row[field] is not None]
                summary[f"{field}_mean"] = float(np.mean(values)) if values else None
                summary[f"{field}_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary["bias_dominant_seed_count"] = sum(row["bias_dominant_flag"] for row in selected)
            summary["magnitude_mismatch_seed_count"] = sum(row["magnitude_mismatch_flag"] for row in selected)
            summary["poor_direction_seed_count"] = sum(row["poor_direction_flag"] for row in selected)
            summaries.append(summary)

    if not all(sha256(ROOT / relative) == digest for relative, digest in before.items()):
        raise RuntimeError("protected input/config/checkpoint/result was modified")

    output.mkdir(parents=True)
    write_csv(output / "per_seed_bearing_metrics.csv", metric_rows)
    write_csv(output / "per_dimension_metrics.csv", dimension_rows)
    write_csv(output / "per_target_metrics.csv", target_rows)
    write_csv(output / "summary_by_bearing.csv", summaries)
    write_csv(output / "integrity_checks.csv", integrity_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {"control": "#3274A1", "ranking": "#E1812C"}
    x = np.arange(len(cfg["bearings"]))
    for arm in cfg["arms"]:
        selected = [next(row for row in summaries if row["arm"] == arm and row["bearing_id"] == bearing) for bearing in cfg["bearings"]]
        axes[0].plot(x, [row["raw_mse_ratio_mean"] for row in selected], marker="o", color=colors[arm], label=f"{arm}: raw")
        axes[0].plot(x, [row["oracle_bias_corrected_mse_ratio_mean"] for row in selected], marker="s", linestyle="--", color=colors[arm], label=f"{arm}: oracle debiased")
        axes[1].plot(x, [row["magnitude_ratio_mean"] for row in selected], marker="o", color=colors[arm], label=arm)
        axes[2].plot(x, [row["flattened_cosine_mean"] for row in selected], marker="o", color=colors[arm], label=arm)
    labels = [bearing.replace("Bearing", "B") for bearing in cfg["bearings"]]
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].axhline(1, color="black", linestyle=":")
    axes[0].set_yscale("log")
    axes[0].set_title("Forecast error ratio")
    axes[0].set_ylabel("MSE / persistence MSE (log)")
    axes[1].axhline(1, color="black", linestyle=":")
    axes[1].set_title("Increment magnitude")
    axes[1].set_ylabel("predicted RMS / true RMS")
    axes[2].axhline(0, color="black", linestyle=":")
    axes[2].set_title("Increment direction")
    axes[2].set_ylabel("flattened cosine")
    fig.suptitle("Frozen GRU increment diagnosis; mean across three seeds")
    fig.savefig(output / "increment_diagnosis.png", dpi=170)
    plt.close(fig)

    report = {
        "status": "completed",
        "verification_status": "exact_persistence_mse_reproduction_and_protected_files_unchanged",
        "config": cfg,
        "protected_sha256": before,
        "protected_files_unchanged": True,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                Path(__file__), ROOT / "src/state_interpreter/increment_diagnostics.py",
                ROOT / "src/state_interpreter/gru_temporal.py",
                ROOT / "scripts/run_gru_interpreter_exploratory.py",
            )
        },
        "environment": {
            "executable": sys.executable, "python": sys.version,
            "torch": str(torch.__version__), "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "oracle_bias_correction_is_deployable": False,
        "duration_seconds": time.monotonic() - started,
        "row_counts": {
            "per_seed_bearing_metrics": len(metric_rows),
            "per_dimension_metrics": len(dimension_rows),
            "per_target_metrics": len(target_rows),
            "summary_by_bearing": len(summaries),
            "integrity_checks": len(integrity_rows),
        },
        "max_archived_mse_difference": max(row["archived_mse_absolute_difference"] for row in integrity_rows),
        "max_increment_vs_direct_mse_difference": max(row["increment_vs_direct_mse_absolute_difference"] for row in integrity_rows),
        "max_prefix_difference": max(row["prefix_max_absolute_difference"] for row in integrity_rows),
        "summaries": summaries,
        "fallacy_scan_coverage": "11/11",
        "limitations": [
            "Already inspected bearings; this is not a new blind test.",
            "Bearing1_4 influenced checkpoint selection and Bearing1_5 has been repeatedly inspected.",
            "Three seeds quantify training randomness, not between-device uncertainty.",
            "Oracle bias correction uses the complete evaluated target sequence and is not deployable.",
            "Descriptive thresholds are not inferential hypothesis tests.",
            "Increment forecast quality does not prove physical health semantics or decision utility.",
        ],
    }
    write_json(output / "experiment_report.json", report)
    print(json.dumps({
        "status": "completed", "row_counts": report["row_counts"],
        "max_archived_mse_difference": report["max_archived_mse_difference"],
        "max_increment_vs_direct_mse_difference": report["max_increment_vs_direct_mse_difference"],
        "summaries": summaries,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
