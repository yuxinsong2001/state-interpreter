"""Compare HMMs fitted on raw and bearing-centered latent trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
repository_src = repository_root / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import numpy as np
import torch
from scipy.stats import spearmanr

from state_interpreter import LeftRightGaussianHMMStateInterpreter


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
        if len(steps) != len(set(steps)) or steps != sorted(steps):
            raise ValueError(f"invalid step order for {bearing}")
    return by_bearing, latent_columns


def to_latent(rows: list[dict[str, str]], columns: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=torch.float64,
    )


def center_on_early_reference(latent: torch.Tensor, calibration_steps: int) -> torch.Tensor:
    if latent.ndim != 2:
        raise ValueError("latent sequence must have shape [time, embedding_dim]")
    if calibration_steps < 1 or calibration_steps > len(latent):
        raise ValueError("calibration_steps must be within the sequence length")
    reference = latent[:calibration_steps].mean(dim=0, keepdim=True)
    centered = latent - reference
    if not torch.allclose(
        centered[:calibration_steps].mean(dim=0),
        torch.zeros(latent.shape[1], dtype=latent.dtype),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("early-reference centering invariant failed")
    return centered


def trajectory_metrics(
    normalized_lifetime: np.ndarray,
    expected_stage: np.ndarray,
    discrete_stage: np.ndarray,
    confidence: np.ndarray,
    score_mask: np.ndarray,
) -> dict[str, object]:
    values = expected_stage[score_mask]
    stages = discrete_stage[score_mask]
    selected_confidence = confidence[score_mask]
    differences = np.diff(values)
    state_counts = [int(np.count_nonzero(stages == state)) for state in range(3)]
    return {
        "score_samples": int(len(values)),
        "spearman_rho": safe_spearman(normalized_lifetime[score_mask], values),
        "first_last_delta": float(values[-1] - values[0]),
        "mean_absolute_step": float(np.mean(np.abs(differences))),
        "step_std": float(np.std(differences)),
        "backward_step_fraction": float(np.mean(differences < -1e-9)),
        "stage_changes": int(np.count_nonzero(np.diff(stages))),
        "mean_confidence": float(np.mean(selected_confidence)),
        "stage_0_count": state_counts[0],
        "stage_1_count": state_counts[1],
        "stage_2_count": state_counts[2],
        "collapsed_to_one_stage": len(set(stages.tolist())) == 1,
    }


def build_hmm(config: dict[str, object], embedding_dim: int) -> LeftRightGaussianHMMStateInterpreter:
    return LeftRightGaussianHMMStateInterpreter(
        embedding_dim=embedding_dim,
        num_states=int(config["num_states"]),
        max_iterations=int(config["max_iterations"]),
        tolerance=float(config["tolerance"]),
        variance_floor=float(config["variance_floor"]),
        transition_floor=float(config["transition_floor"]),
    )


def serialize_parameters(hmm: LeftRightGaussianHMMStateInterpreter) -> dict[str, object]:
    return {
        "feature_mean": hmm.feature_mean_.tolist(),
        "feature_std": hmm.feature_std_.tolist(),
        "emission_means_standardized": hmm.means_.tolist(),
        "emission_variances_standardized": hmm.variances_.tolist(),
        "transition_matrix": hmm.transition_matrix_.tolist(),
        "iterations": hmm.n_iterations_,
        "log_likelihood_history": hmm.log_likelihood_history_,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (repository_root / config["input_latents"]).resolve()
    output_dir = (repository_root / config["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()

    by_bearing, latent_columns = load_latents(input_path)
    raw_latents = {
        bearing: to_latent(rows, latent_columns)
        for bearing, rows in by_bearing.items()
    }
    calibration_steps = int(config["calibration_steps"])
    centered_latents = {
        bearing: center_on_early_reference(latent, calibration_steps)
        for bearing, latent in raw_latents.items()
    }
    split_by_bearing = {
        bearing: rows[0]["split"] for bearing, rows in by_bearing.items()
    }
    train_bearings = sorted(
        bearing
        for bearing, split in split_by_bearing.items()
        if split == config["training_split"]
    )
    if len(train_bearings) < 2:
        raise ValueError("at least two training bearing sequences are required")

    raw_hmm = build_hmm(config, len(latent_columns)).fit(
        [raw_latents[bearing] for bearing in train_bearings]
    )
    centered_hmm = build_hmm(config, len(latent_columns)).fit(
        [centered_latents[bearing] for bearing in train_bearings]
    )

    models = {"raw_z": raw_hmm, "bearing_centered_z": centered_hmm}
    representations = {"raw_z": raw_latents, "bearing_centered_z": centered_latents}
    per_step_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    for bearing in sorted(by_bearing):
        rows = by_bearing[bearing]
        normalized_lifetime = np.asarray(
            [float(row["normalized_lifetime"]) for row in rows], dtype=float
        )
        score_mask = np.arange(len(rows)) >= int(config["common_score_start_step"])
        if int(score_mask.sum()) < 3:
            raise ValueError(f"too few common scoring samples for {bearing}")
        outputs_by_representation = {}
        metrics_by_representation = {}
        for representation_name in ("raw_z", "bearing_centered_z"):
            outputs = models[representation_name].transform(
                representations[representation_name][bearing]
            )
            expected_stage = np.asarray(
                [float(output.expected_stage) for output in outputs], dtype=float
            )
            discrete_stage = np.asarray([int(output.stage) for output in outputs])
            confidence = np.asarray(
                [float(output.confidence) for output in outputs], dtype=float
            )
            metrics = trajectory_metrics(
                normalized_lifetime,
                expected_stage,
                discrete_stage,
                confidence,
                score_mask,
            )
            metric_rows.append(
                {
                    "bearing_id": bearing,
                    "split": split_by_bearing[bearing],
                    "representation": representation_name,
                    **metrics,
                }
            )
            outputs_by_representation[representation_name] = outputs
            metrics_by_representation[representation_name] = metrics

        raw_rho = float(metrics_by_representation["raw_z"]["spearman_rho"])
        centered_rho = float(
            metrics_by_representation["bearing_centered_z"]["spearman_rho"]
        )
        comparison_rows.append(
            {
                "bearing_id": bearing,
                "split": split_by_bearing[bearing],
                "raw_spearman_rho": raw_rho,
                "centered_spearman_rho": centered_rho,
                "rho_delta_centered_minus_raw": (
                    centered_rho - raw_rho
                    if math.isfinite(raw_rho) and math.isfinite(centered_rho)
                    else float("nan")
                ),
                "raw_stage_changes": metrics_by_representation["raw_z"]["stage_changes"],
                "centered_stage_changes": metrics_by_representation["bearing_centered_z"]["stage_changes"],
                "raw_collapsed": metrics_by_representation["raw_z"]["collapsed_to_one_stage"],
                "centered_collapsed": metrics_by_representation["bearing_centered_z"]["collapsed_to_one_stage"],
            }
        )
        for index, row in enumerate(rows):
            output_row: dict[str, object] = {
                "bearing_id": bearing,
                "split": split_by_bearing[bearing],
                "step_id": int(row["step_id"]),
                "normalized_lifetime": normalized_lifetime[index],
                "included_in_common_score": bool(score_mask[index]),
            }
            for representation_name, prefix in (
                ("raw_z", "raw"),
                ("bearing_centered_z", "centered"),
            ):
                output = outputs_by_representation[representation_name][index]
                output_row[f"{prefix}_expected_stage"] = float(output.expected_stage)
                output_row[f"{prefix}_stage"] = int(output.stage)
                output_row[f"{prefix}_confidence"] = float(output.confidence)
                for state, probability in enumerate(output.stage_probabilities.tolist()):
                    output_row[f"{prefix}_probability_{state}"] = probability
            per_step_rows.append(output_row)

    for filename, rows in (
        ("per_step_states.csv", per_step_rows),
        ("bearing_metrics.csv", metric_rows),
        ("representation_comparison.csv", comparison_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    parameters = {
        name: serialize_parameters(model) for name, model in models.items()
    }
    (output_dir / "hmm_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "status": "completed",
        "verification_status": "exploratory_ablation",
        "experiment_id": config["experiment_id"],
        "hypothesis": (
            "Per-bearing early-reference centering reduces cross-bearing initial-position "
            "offset and prevents HMM state collapse without changing the HMM structure."
        ),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "train_bearings": train_bearings,
        "evaluated_bearings": sorted(by_bearing),
        "calibration_steps": calibration_steps,
        "common_score_start_step": int(config["common_score_start_step"]),
        "controlled_factors": [
            "same archived AutoEncoder embeddings",
            "same training-bearing split",
            "same three-state left-right diagonal-Gaussian HMM",
            "same EM hyperparameters and causal log-domain filtering",
            "same scoring interval",
        ],
        "changed_factor": (
            "subtract each bearing's own first-15-embedding mean before HMM fitting and inference"
        ),
        "comparisons": comparison_rows,
        "warnings": [
            "All evaluated bearings were previously inspected; this is not a blind test.",
            "Normalized lifetime is a time proxy, not a physical damage label.",
            "Training-bearing metrics are descriptive and not generalization evidence.",
            "The first 15 measurements are assumed to provide a local reference; this assumption is not physically labeled.",
            "Only centering is tested here; bearing-specific scale normalization is deliberately excluded.",
        ],
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "per_step_states.csv",
            "bearing_metrics.csv",
            "representation_comparison.csv",
            "hmm_parameters.json",
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
