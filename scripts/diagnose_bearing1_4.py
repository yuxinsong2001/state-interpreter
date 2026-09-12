"""Post-hoc diagnosis of the weakest v0.1 stability-study bearing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
repository_src = repository_root / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import numpy as np
import torch

from run_stability_study import interpreter_metrics, make_folds, train_model
from state_interpreter import RelativeTemporalStateInterpreter
from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer, materialize_stft_data
from state_interpreter.preprocessing import LogSTFTPreprocessor


SEEDS = (20260801, 20260802, 20260803)
CALIBRATIONS = (5, 10, 15, 20)
WINDOWS = (3, 5, 10, 20)


def level_series(latent: torch.Tensor, calibration: int, window: int) -> np.ndarray:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=latent.shape[1],
        calibration_steps=calibration,
        temporal_window=window,
    )
    levels = []
    for z in latent:
        output = interpreter.update(z)
        if output is not None:
            levels.append(float(output.level))
    return np.asarray(levels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    condition = "35Hz12kN"
    bearings = tuple(f"Bearing1_{index}" for index in range(1, 6))
    fold = next(fold for fold in make_folds(bearings) if fold.test == "Bearing1_4")
    preprocessor = LogSTFTPreprocessor()
    by_bearing = {}
    for bearing in bearings:
        adapter = XJTUSYDatasetAdapter(args.root, conditions=[condition], bearings=[bearing])
        by_bearing[bearing] = materialize_stft_data(adapter, preprocessor).tensors

    raw_train = torch.cat([by_bearing[bearing] for bearing in fold.train])
    standardizer = ChannelStandardizer.fit(raw_train)
    train = standardizer.transform(raw_train)
    validation = standardizer.transform(by_bearing[fold.validation])
    standardized = {
        bearing: standardizer.transform(tensors) for bearing, tensors in by_bearing.items()
    }

    reconstruction_rows = []
    sensitivity_rows = []
    baseline_levels = {}
    for seed in SEEDS:
        print(f"TRAIN seed={seed}", flush=True)
        model, best_epoch, validation_loss = train_model(
            train,
            validation,
            seed=seed,
            latent_dim=8,
            epochs=5,
            batch_size=32,
            learning_rate=1e-3,
        )
        with torch.inference_mode():
            outputs = {bearing: model(values) for bearing, values in standardized.items()}
        for bearing, output in outputs.items():
            reconstruction_rows.append(
                {
                    "seed": seed,
                    "bearing": bearing,
                    "role": "train" if bearing in fold.train else "validation" if bearing == fold.validation else "test",
                    "samples": len(standardized[bearing]),
                    "best_epoch": best_epoch,
                    "validation_mse": validation_loss,
                    "reconstruction_mse": float(
                        (output.reconstruction - standardized[bearing]).square().mean()
                    ),
                }
            )
        latent = outputs[fold.test].z
        baseline_levels[seed] = level_series(latent, 15, 10)
        for calibration in CALIBRATIONS:
            for window in WINDOWS:
                sensitivity_rows.append(
                    {
                        "seed": seed,
                        "calibration_steps": calibration,
                        "temporal_window": window,
                        **interpreter_metrics(latent, calibration, window),
                    }
                )

    pairwise = []
    for index, left_seed in enumerate(SEEDS):
        for right_seed in SEEDS[index + 1 :]:
            left = baseline_levels[left_seed]
            right = baseline_levels[right_seed]
            pairwise.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "level_trajectory_pearson": float(np.corrcoef(left, right)[0, 1]),
                }
            )

    for filename, rows in (
        ("reconstruction_by_bearing.csv", reconstruction_rows),
        ("window_sensitivity.csv", sensitivity_rows),
        ("seed_trajectory_agreement.csv", pairwise),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    target_mse = np.mean(
        [row["reconstruction_mse"] for row in reconstruction_rows if row["bearing"] == fold.test]
    )
    other_mse = np.mean(
        [row["reconstruction_mse"] for row in reconstruction_rows if row["bearing"] != fold.test]
    )
    locked = [
        row for row in sensitivity_rows
        if row["calibration_steps"] == 15 and row["temporal_window"] == 10
    ]
    by_setting = {}
    for calibration in CALIBRATIONS:
        for window in WINDOWS:
            rows = [
                row for row in sensitivity_rows
                if row["calibration_steps"] == calibration and row["temporal_window"] == window
            ]
            by_setting[f"{calibration}/{window}"] = {
                "level_rho_mean": float(np.mean([row["level_spearman_rho"] for row in rows])),
                "trend_accuracy_mean": float(np.mean([row["trend_direction_accuracy"] for row in rows])),
                "movement_rho_mean": float(np.mean([row["movement_change_spearman_rho"] for row in rows])),
            }
    report = {
        "status": "completed",
        "analysis_type": "post_hoc_failure_diagnosis_not_for_retuning_v0.1",
        "condition": condition,
        "fold": {"train": fold.train, "validation": fold.validation, "test": fold.test},
        "seeds": SEEDS,
        "reconstruction": {
            "bearing1_4_mean_mse": float(target_mse),
            "other_bearings_mean_mse": float(other_mse),
            "ratio": float(target_mse / other_mse),
        },
        "locked_15_10": {
            "level_rho_mean": float(np.mean([row["level_spearman_rho"] for row in locked])),
            "level_rho_min": float(np.min([row["level_spearman_rho"] for row in locked])),
            "level_rho_max": float(np.max([row["level_spearman_rho"] for row in locked])),
        },
        "seed_level_trajectory_pearson_mean": float(
            np.mean([row["level_trajectory_pearson"] for row in pairwise])
        ),
        "setting_means": by_setting,
        "interpretation_rule": (
            "High reconstruction ratio suggests encoder/domain mismatch; low seed agreement suggests "
            "initialization instability; large setting spread suggests window sensitivity."
        ),
        "warning": "Exploratory diagnosis on a seen test bearing; results must not retune or relabel v0.1.",
    }
    with (output_dir / "diagnostic_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
