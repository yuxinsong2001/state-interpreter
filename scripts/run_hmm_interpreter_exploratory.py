"""Compare the fixed distance baseline with a causal left-right HMM."""

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

from state_interpreter import (
    LeftRightGaussianHMMStateInterpreter,
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
        if len(steps) != len(set(steps)) or steps != sorted(steps):
            raise ValueError(f"invalid step order for {bearing}")
    return by_bearing, latent_columns


def to_latent(rows: list[dict[str, str]], columns: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=torch.float64,
    )


def baseline_level(
    latent: torch.Tensor, calibration_steps: int, temporal_window: int
) -> np.ndarray:
    model = RelativeTemporalStateInterpreter(
        embedding_dim=latent.shape[1],
        calibration_steps=calibration_steps,
        temporal_window=temporal_window,
    )
    values = np.full(len(latent), np.nan, dtype=float)
    for index, z in enumerate(latent.to(torch.float32)):
        output = model.update(z)
        if output is not None:
            values[index] = float(output.level)
    return values


def trajectory_metrics(
    normalized_lifetime: np.ndarray,
    values: np.ndarray,
    score_mask: np.ndarray,
) -> dict[str, float | int]:
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
    latents = {
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
    if len(train_bearings) < 2:
        raise ValueError("at least two training bearing sequences are required")

    hmm = LeftRightGaussianHMMStateInterpreter(
        embedding_dim=len(latent_columns),
        num_states=int(config["num_states"]),
        max_iterations=int(config["max_iterations"]),
        tolerance=float(config["tolerance"]),
        variance_floor=float(config["variance_floor"]),
        transition_floor=float(config["transition_floor"]),
    ).fit([latents[bearing] for bearing in train_bearings])

    per_step_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for bearing in sorted(by_bearing):
        rows = by_bearing[bearing]
        latent = latents[bearing]
        normalized_lifetime = np.asarray(
            [float(row["normalized_lifetime"]) for row in rows], dtype=float
        )
        baseline = baseline_level(
            latent,
            int(config["baseline_calibration_steps"]),
            int(config["baseline_temporal_window"]),
        )
        hmm_outputs = hmm.transform(latent)
        hmm_expected = np.asarray(
            [float(output.expected_stage) for output in hmm_outputs], dtype=float
        )
        hmm_stage = np.asarray([int(output.stage) for output in hmm_outputs])
        hmm_confidence = np.asarray(
            [float(output.confidence) for output in hmm_outputs], dtype=float
        )
        common_mask = np.arange(len(rows)) >= int(config["common_score_start_step"])
        common_mask &= np.isfinite(baseline)
        if int(common_mask.sum()) < 3:
            raise ValueError(f"too few common scoring samples for {bearing}")
        for method, values in (("distance_baseline", baseline), ("hmm", hmm_expected)):
            metric_rows.append(
                {
                    "bearing_id": bearing,
                    "split": split_by_bearing[bearing],
                    "method": method,
                    **trajectory_metrics(normalized_lifetime, values, common_mask),
                    "stage_changes": (
                        int(np.count_nonzero(np.diff(hmm_stage[common_mask])))
                        if method == "hmm"
                        else ""
                    ),
                    "mean_confidence": (
                        float(np.mean(hmm_confidence[common_mask]))
                        if method == "hmm"
                        else ""
                    ),
                }
            )
        for index, row in enumerate(rows):
            probabilities = hmm_outputs[index].stage_probabilities.tolist()
            output_row: dict[str, object] = {
                "bearing_id": bearing,
                "split": split_by_bearing[bearing],
                "step_id": int(row["step_id"]),
                "normalized_lifetime": normalized_lifetime[index],
                "distance_level": "" if math.isnan(baseline[index]) else baseline[index],
                "hmm_expected_stage": hmm_expected[index],
                "hmm_stage": hmm_stage[index],
                "hmm_confidence": hmm_confidence[index],
                "included_in_common_score": bool(common_mask[index]),
            }
            output_row.update(
                {f"hmm_probability_{state}": value for state, value in enumerate(probabilities)}
            )
            per_step_rows.append(output_row)
    with (output_dir / "per_step_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_step_rows[0]))
        writer.writeheader()
        writer.writerows(per_step_rows)
    with (output_dir / "bearing_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    parameters = {
        "feature_mean": hmm.feature_mean_.tolist(),
        "feature_std": hmm.feature_std_.tolist(),
        "emission_means_standardized": hmm.means_.tolist(),
        "emission_variances_standardized": hmm.variances_.tolist(),
        "transition_matrix": hmm.transition_matrix_.tolist(),
        "iterations": hmm.n_iterations_,
        "log_likelihood_history": hmm.log_likelihood_history_,
    }
    (output_dir / "hmm_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "status": "completed",
        "verification_status": "unverified_exploratory",
        "experiment_id": config["experiment_id"],
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "train_bearings": train_bearings,
        "evaluated_bearings": sorted(by_bearing),
        "common_score_start_step": int(config["common_score_start_step"]),
        "hmm": {
            "num_states": hmm.num_states,
            "inference": "causal filtered probabilities using current and past embeddings",
            "fitting": "EM on complete training sequences with diagonal Gaussian emissions",
            "state_order": "relative-time initialization plus left-to-right self/next transitions",
        },
        "metrics": metric_rows,
        "warnings": [
            "All evaluated bearings were previously inspected; this is not a blind test.",
            "Normalized lifetime is a time proxy, not a physical damage label.",
            "Training-bearing metrics are descriptive and not generalization evidence.",
            "HMM state numbers are temporal stages; health semantics are not guaranteed.",
        ],
        "duration_seconds": time.monotonic() - started,
        "outputs": [
            "per_step_states.csv",
            "bearing_metrics.csv",
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
