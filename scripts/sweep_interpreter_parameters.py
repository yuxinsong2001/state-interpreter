"""Validation-only sensitivity sweep for the online State Interpreter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

repository_src = Path(__file__).resolve().parents[1] / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from state_interpreter import RelativeTemporalStateInterpreter


DEFAULT_TRAIN_BEARINGS = ("Bearing1_1", "Bearing1_2", "Bearing1_3")
DEFAULT_VALIDATION_BEARING = "Bearing1_4"


def parse_grid(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("parameter grid must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("parameter grid must not contain duplicates")
    return values


def select_development_rows(
    rows: list[dict[str, str]],
    train_bearings: tuple[str, ...],
    validation_bearing: str,
) -> tuple[list[dict[str, str]], list[str]]:
    allowed = set(train_bearings) | {validation_bearing}
    selected = [row for row in rows if row["bearing_id"] in allowed]
    excluded = sorted({row["bearing_id"] for row in rows if row["bearing_id"] not in allowed})
    present = {row["bearing_id"] for row in selected}
    missing = allowed - present
    if missing:
        raise ValueError(f"development bearings missing from input: {sorted(missing)}")
    return selected, excluded


def directional_monotonicity(values: np.ndarray) -> float:
    differences = np.diff(values)
    nonzero = np.sign(differences[np.abs(differences) > 1e-12])
    return 0.0 if len(nonzero) == 0 else float(nonzero.mean())


def roughness_smoothness(values: np.ndarray) -> float:
    first = np.diff(values)
    if len(first) < 2:
        return 1.0
    roughness = np.sum(np.abs(np.diff(first))) / (np.sum(np.abs(first)) + 1e-12)
    return float(1.0 / (1.0 + roughness))


def replay(
    rows: list[dict[str, str]],
    z_fields: list[str],
    calibration_steps: int,
    temporal_window: int,
) -> dict[str, np.ndarray | int | float]:
    if len(rows) <= calibration_steps:
        raise ValueError("calibration consumes the complete episode")
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=len(z_fields),
        calibration_steps=calibration_steps,
        temporal_window=temporal_window,
    )
    lifetimes: list[float] = []
    levels: list[float] = []
    trends: list[float] = []
    movements: list[float] = []
    for row in rows:
        z = torch.tensor([float(row[field]) for field in z_fields])
        output = interpreter.update(z)
        if output is None:
            continue
        lifetimes.append(float(row["normalized_lifetime"]))
        levels.append(float(output.level))
        trends.append(float(output.trend))
        movements.append(float(output.movement))
    level_array = np.asarray(levels)
    trend_array = np.asarray(trends)
    movement_array = np.asarray(movements)
    rho = float(spearmanr(lifetimes, levels).statistic)
    return {
        "ready_samples": len(levels),
        "ready_fraction": len(levels) / len(rows),
        "level_spearman_rho": rho,
        "level_directional_monotonicity": directional_monotonicity(level_array),
        "level_smoothness": roughness_smoothness(level_array),
        "trend_noise": float(np.std(np.diff(trend_array))) if len(trend_array) > 1 else 0.0,
        "movement_max": float(movement_array.max()),
        "movement_last": float(movement_array[-1]),
        "level_first": float(level_array[0]),
        "level_last": float(level_array[-1]),
    }


def save_heatmap(
    path: Path,
    results: list[dict[str, object]],
    calibration_grid: tuple[int, ...],
    window_grid: tuple[int, ...],
    metric: str,
    title: str,
) -> None:
    matrix = np.empty((len(calibration_grid), len(window_grid)))
    for row_index, calibration_steps in enumerate(calibration_grid):
        for column_index, temporal_window in enumerate(window_grid):
            match = next(
                row
                for row in results
                if row["calibration_steps"] == calibration_steps
                and row["temporal_window"] == temporal_window
            )
            matrix[row_index, column_index] = float(match[metric])
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(window_grid)), labels=window_grid)
    axis.set_yticks(range(len(calibration_grid)), labels=calibration_grid)
    axis.set_xlabel("Temporal window")
    axis.set_ylabel("Calibration steps")
    axis.set_title(title)
    for row_index in range(len(calibration_grid)):
        for column_index in range(len(window_grid)):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] < matrix.mean() else "black",
            )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-grid", default="5,10,15,20")
    parser.add_argument("--window-grid", default="3,5,10")
    parser.add_argument("--validation-bearing", default=DEFAULT_VALIDATION_BEARING)
    args = parser.parse_args()

    calibration_grid = parse_grid(args.calibration_grid)
    window_grid = parse_grid(args.window_grid)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    with Path(args.input).open("r", newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    development_rows, excluded_bearings = select_development_rows(
        all_rows, DEFAULT_TRAIN_BEARINGS, args.validation_bearing
    )
    z_fields = sorted(
        (name for name in development_rows[0] if name.startswith("z_")),
        key=lambda name: int(name.split("_")[1]),
    )
    bearing_order = (*DEFAULT_TRAIN_BEARINGS, args.validation_bearing)
    by_bearing = {
        bearing: [row for row in development_rows if row["bearing_id"] == bearing]
        for bearing in bearing_order
    }

    per_bearing: list[dict[str, object]] = []
    configurations: list[dict[str, object]] = []
    for calibration_steps in calibration_grid:
        for temporal_window in window_grid:
            metrics_by_bearing: dict[str, dict[str, np.ndarray | int | float]] = {}
            for bearing in bearing_order:
                metrics = replay(
                    by_bearing[bearing], z_fields, calibration_steps, temporal_window
                )
                metrics_by_bearing[bearing] = metrics
                per_bearing.append(
                    {
                        "calibration_steps": calibration_steps,
                        "temporal_window": temporal_window,
                        "bearing_id": bearing,
                        "split": "validation" if bearing == args.validation_bearing else "train",
                        **metrics,
                    }
                )
            validation = metrics_by_bearing[args.validation_bearing]
            train_rhos = np.asarray(
                [metrics_by_bearing[bearing]["level_spearman_rho"] for bearing in DEFAULT_TRAIN_BEARINGS],
                dtype=float,
            )
            configurations.append(
                {
                    "calibration_steps": calibration_steps,
                    "temporal_window": temporal_window,
                    "first_ready_step": calibration_steps,
                    "full_window_step": calibration_steps + temporal_window - 1,
                    "train_mean_spearman_rho": float(train_rhos.mean()),
                    "train_min_spearman_rho": float(train_rhos.min()),
                    "validation_spearman_rho": validation["level_spearman_rho"],
                    "validation_directional_monotonicity": validation[
                        "level_directional_monotonicity"
                    ],
                    "validation_smoothness": validation["level_smoothness"],
                    "validation_trend_noise": validation["trend_noise"],
                    "validation_ready_samples": validation["ready_samples"],
                    "validation_ready_fraction": validation["ready_fraction"],
                    "validation_movement_last": validation["movement_last"],
                    "validation_movement_max": validation["movement_max"],
                }
            )

    ranked = sorted(
        configurations,
        key=lambda row: (
            -float(row["validation_spearman_rho"]),
            -float(row["validation_smoothness"]),
            int(row["full_window_step"]),
        ),
    )
    best_rho = float(ranked[0]["validation_spearman_rho"])
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["within_0_03_of_best_validation_rho"] = (
            best_rho - float(row["validation_spearman_rho"]) <= 0.03
        )

    configuration_fields = ["rank"] + [
        name for name in ranked[0] if name != "rank"
    ]
    with (output_dir / "sweep_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=configuration_fields)
        writer.writeheader()
        writer.writerows(ranked)
    with (output_dir / "per_bearing_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_bearing[0]))
        writer.writeheader()
        writer.writerows(per_bearing)

    save_heatmap(
        output_dir / "validation_spearman_heatmap.png",
        configurations,
        calibration_grid,
        window_grid,
        "validation_spearman_rho",
        f"Validation Spearman rho ({args.validation_bearing})",
    )
    save_heatmap(
        output_dir / "validation_smoothness_heatmap.png",
        configurations,
        calibration_grid,
        window_grid,
        "validation_smoothness",
        f"Validation smoothness ({args.validation_bearing})",
    )

    report = {
        "status": "completed",
        "input": str(Path(args.input).resolve()),
        "calibration_grid": calibration_grid,
        "window_grid": window_grid,
        "configuration_count": len(configurations),
        "train_bearings": DEFAULT_TRAIN_BEARINGS,
        "validation_bearing": args.validation_bearing,
        "excluded_bearings": excluded_bearings,
        "holdout_metrics_computed": False,
        "ranking_rule": (
            "validation Spearman rho descending, validation smoothness descending, "
            "full-window availability ascending"
        ),
        "near_best_rule": "validation Spearman rho within 0.03 of the best",
        "top_configuration": ranked[0],
        "near_best_configurations": [
            row for row in ranked if row["within_0_03_of_best_validation_rho"]
        ],
        "metric_warning": (
            "normalized lifetime is a time proxy; sensitivity results do not "
            "establish physical health validity"
        ),
        "interpretation_gate": {
            "passed": False,
            "reason": "Sweep completed; robust-region selection remains to be interpreted.",
        },
    }
    with (output_dir / "sweep_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
