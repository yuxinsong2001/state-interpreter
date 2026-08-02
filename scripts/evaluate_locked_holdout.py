"""Evaluate one locked interpreter configuration on its designated holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for field in ("embedding_dim", "calibration_steps", "temporal_window"):
        if not isinstance(config.get(field), int) or int(config[field]) <= 0:
            raise ValueError(f"locked config has invalid {field}")
    holdout = config.get("holdout")
    if not isinstance(holdout, dict) or not holdout.get("bearing"):
        raise ValueError("locked config has no designated holdout bearing")
    if holdout.get("evaluated_during_selection") is not False:
        raise ValueError("holdout was not isolated during parameter selection")
    if isinstance(config.get("final_evaluation"), dict) and config["final_evaluation"].get(
        "evaluated"
    ):
        raise ValueError("locked configuration already has a final evaluation")
    return config


def select_holdout_rows(
    rows: list[dict[str, str]], holdout_bearing: str
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["bearing_id"] == holdout_bearing]
    if not selected:
        raise ValueError(f"holdout bearing not found: {holdout_bearing}")
    if {row["bearing_id"] for row in selected} != {holdout_bearing}:
        raise RuntimeError("holdout selection contains another bearing")
    steps = [int(row["step_id"]) for row in selected]
    if any(next_step <= step for step, next_step in zip(steps, steps[1:])):
        raise ValueError("holdout steps are not strictly increasing")
    return selected


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config = load_locked_config(config_path)
    config_hash = sha256(config_path)

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    holdout_bearing = str(config["holdout"]["bearing"])
    rows = select_holdout_rows(all_rows, holdout_bearing)
    z_fields = sorted(
        (name for name in rows[0] if name.startswith("z_")),
        key=lambda name: int(name.split("_")[1]),
    )
    if len(z_fields) != int(config["embedding_dim"]):
        raise ValueError("latent dimension does not match locked configuration")

    output_dir.mkdir(parents=True)
    shutil.copyfile(config_path, output_dir / "locked_config_snapshot.json")
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=int(config["embedding_dim"]),
        calibration_steps=int(config["calibration_steps"]),
        temporal_window=int(config["temporal_window"]),
    )
    output_rows: list[dict[str, object]] = []
    ready_lifetime: list[float] = []
    levels: list[float] = []
    trends: list[float] = []
    movements: list[float] = []
    ready_steps: list[int] = []
    for row in rows:
        z = torch.tensor([float(row[field]) for field in z_fields])
        output = interpreter.update(z)
        ready = output is not None
        output_rows.append(
            {
                "bearing_id": holdout_bearing,
                "step_id": int(row["step_id"]),
                "normalized_lifetime": float(row["normalized_lifetime"]),
                "phase": "ready" if ready else "calibrating",
                "level": "" if output is None else float(output.level),
                "trend": "" if output is None else float(output.trend),
                "movement": "" if output is None else float(output.movement),
            }
        )
        if output is not None:
            ready_steps.append(int(row["step_id"]))
            ready_lifetime.append(float(row["normalized_lifetime"]))
            levels.append(float(output.level))
            trends.append(float(output.trend))
            movements.append(float(output.movement))

    level_array = np.asarray(levels)
    trend_array = np.asarray(trends)
    movement_array = np.asarray(movements)
    rho = float(spearmanr(ready_lifetime, levels).statistic)
    validation_rho = float(config["selection_metrics"]["validation_spearman_rho"])
    movement_max_index = int(np.argmax(movement_array))
    metrics = {
        "level_spearman_rho": rho,
        "level_directional_monotonicity": directional_monotonicity(level_array),
        "level_smoothness": roughness_smoothness(level_array),
        "trend_noise": float(np.std(np.diff(trend_array))),
        "ready_samples": len(levels),
        "ready_fraction": len(levels) / len(rows),
        "level_first": float(level_array[0]),
        "level_last": float(level_array[-1]),
        "movement_max": float(movement_array[movement_max_index]),
        "movement_max_step": ready_steps[movement_max_index],
        "movement_last": float(movement_array[-1]),
        "rho_difference_from_validation": rho - validation_rho,
        "direction_consistent": rho > 0.0,
        "within_0_20_below_validation": rho >= validation_rho - 0.20,
    }
    with (output_dir / "holdout_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "status": "completed",
        "evaluation_date": "2026-08-02",
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "locked_config": str(config_path.resolve()),
        "locked_config_sha256_before_evaluation": config_hash,
        "evaluated_bearings": [holdout_bearing],
        "other_bearings_evaluated": False,
        "locked_parameters": {
            "embedding_dim": config["embedding_dim"],
            "calibration_steps": config["calibration_steps"],
            "temporal_window": config["temporal_window"],
        },
        "metrics": metrics,
        "scope": {
            "parameter_selection_holdout": True,
            "strict_project_wide_blind_holdout": False,
            "reason": (
                "Bearing1_5 was excluded from the parameter sweep but had been "
                "inspected in earlier exploratory latent analyses."
            ),
        },
        "interpretation_gate": {
            "direction_consistent": metrics["direction_consistent"],
            "within_0_20_below_validation": metrics[
                "within_0_20_below_validation"
            ],
            "parameters_may_be_retuned_after_this_report": False,
        },
    }
    with (output_dir / "holdout_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for axis, values, label in zip(
        axes,
        (levels, trends, movements),
        ("Level", "Trend", "Movement"),
    ):
        axis.plot(ready_steps, values, linewidth=1.8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Measurement step")
    fig.suptitle(
        f"Locked holdout evaluation: {holdout_bearing} "
        f"({config['calibration_steps']}/{config['temporal_window']})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_state_plot.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
