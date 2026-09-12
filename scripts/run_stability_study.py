"""Run seeded leave-one-bearing-out stability experiments on XJTU-SY.

This is a post-v0.1 robustness study. It never edits preregistered configs or
historical run directories. For each condition and fold, one bearing is held
out for test, the next bearing is used for checkpoint selection, and the
remaining three bearings are used for training and normalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import NamedTuple

repository_root = Path(__file__).resolve().parents[1]
repository_src = repository_root / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import numpy as np
import torch
from scipy.stats import spearmanr, t
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from state_interpreter import RelativeTemporalStateInterpreter
from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import ChannelStandardizer, materialize_stft_data
from state_interpreter.encoders import SmallConvAutoEncoder
from state_interpreter.preprocessing import LogSTFTPreprocessor


DEFAULT_CONDITIONS = ("35Hz12kN", "37.5Hz11kN")


class Fold(NamedTuple):
    train: tuple[str, ...]
    validation: str
    test: str


def make_folds(bearings: tuple[str, ...]) -> tuple[Fold, ...]:
    """Return deterministic rotating test/validation folds."""

    if len(bearings) < 3 or len(set(bearings)) != len(bearings):
        raise ValueError("at least three unique bearings are required")
    folds = []
    for test_index, test_bearing in enumerate(bearings):
        validation = bearings[(test_index + 1) % len(bearings)]
        train = tuple(
            bearing for bearing in bearings if bearing not in {test_bearing, validation}
        )
        folds.append(Fold(train=train, validation=validation, test=test_bearing))
    return tuple(folds)


def parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("list must contain unique non-empty values")
    return values


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in parse_csv(value))
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    return seeds


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def interpreter_metrics(latent: torch.Tensor, calibration: int, window: int) -> dict[str, float | int]:
    interpreter = RelativeTemporalStateInterpreter(
        embedding_dim=latent.shape[1],
        calibration_steps=calibration,
        temporal_window=window,
    )
    levels: list[float] = []
    trends: list[float] = []
    movements: list[float] = []
    for z in latent:
        state = interpreter.update(z)
        if state is not None:
            levels.append(float(state.level))
            trends.append(float(state.trend))
            movements.append(float(state.movement))
    if len(levels) < 3:
        raise ValueError("too few READY states for evaluation")
    level = np.asarray(levels)
    trend = np.asarray(trends)
    movement = np.asarray(movements)
    lifetime = np.linspace(0.0, 1.0, len(level), dtype=float)
    level_delta = np.diff(level)
    aligned_trend = trend[1:]
    nonzero = np.abs(level_delta) > 1e-12
    trend_direction_accuracy = (
        float(np.mean(np.sign(aligned_trend[nonzero]) == np.sign(level_delta[nonzero])))
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "ready_samples": len(levels),
        "level_spearman_rho": safe_spearman(lifetime, level),
        "level_first_last_delta": float(level[-1] - level[0]),
        "trend_direction_accuracy": trend_direction_accuracy,
        "trend_noise": float(np.std(np.diff(trend))),
        "movement_change_spearman_rho": safe_spearman(
            movement[1:], np.abs(level_delta)
        ),
        "movement_mean": float(np.mean(movement)),
        "movement_max": float(np.max(movement)),
    }


def confidence_interval(values: list[float]) -> dict[str, float | int]:
    clean = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    margin = float(t.ppf(0.975, len(clean) - 1) * std / math.sqrt(len(clean))) if len(clean) > 1 else 0.0
    return {"n": len(clean), "mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin}


def train_model(
    train: torch.Tensor,
    validation: torch.Tensor,
    *,
    seed: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[SmallConvAutoEncoder, int, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train), batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(TensorDataset(validation), batch_size=batch_size, shuffle=False)
    model = SmallConvAutoEncoder(input_channels=2, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch).reconstruction, batch)
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for (batch,) in validation_loader:
                loss = loss_fn(model(batch).reconstruction, batch)
                total += float(loss) * len(batch)
                count += len(batch)
        validation_loss = total / count
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--seeds", default="20260801,20260802,20260803")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--calibration-steps", type=int, default=15)
    parser.add_argument("--temporal-window", type=int, default=10)
    args = parser.parse_args()
    conditions = parse_csv(args.conditions)
    seeds = parse_seeds(args.seeds)
    if min(args.epochs, args.batch_size, args.latent_dim, args.calibration_steps, args.temporal_window) <= 0:
        raise ValueError("numeric parameters must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    preprocessor = LogSTFTPreprocessor()
    rows: list[dict[str, object]] = []

    for condition_index, condition in enumerate(conditions, start=1):
        prefix = f"Bearing{condition_index}_"
        bearings = tuple(f"{prefix}{index}" for index in range(1, 6))
        print(f"MATERIALIZING condition={condition}", flush=True)
        by_bearing = {}
        for bearing in bearings:
            adapter = XJTUSYDatasetAdapter(args.root, conditions=[condition], bearings=[bearing])
            by_bearing[bearing] = materialize_stft_data(adapter, preprocessor).tensors
        for fold_index, fold in enumerate(make_folds(bearings), start=1):
            raw_train = torch.cat([by_bearing[bearing] for bearing in fold.train])
            standardizer = ChannelStandardizer.fit(raw_train)
            train = standardizer.transform(raw_train)
            validation = standardizer.transform(by_bearing[fold.validation])
            test = standardizer.transform(by_bearing[fold.test])
            for seed in seeds:
                print(f"RUN condition={condition} fold={fold_index} test={fold.test} seed={seed}", flush=True)
                model, best_epoch, validation_loss = train_model(
                    train, validation, seed=seed, latent_dim=args.latent_dim,
                    epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
                with torch.inference_mode():
                    output = model(test)
                    reconstruction_mse = float((output.reconstruction - test).square().mean())
                rows.append({
                    "condition": condition, "fold": fold_index, "seed": seed,
                    "train_bearings": ";".join(fold.train),
                    "validation_bearing": fold.validation, "test_bearing": fold.test,
                    "best_epoch": best_epoch, "validation_reconstruction_mse": validation_loss,
                    "test_reconstruction_mse": reconstruction_mse,
                    **interpreter_metrics(output.z, args.calibration_steps, args.temporal_window),
                })

    with (output_dir / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "level_spearman_rho", "level_first_last_delta", "trend_direction_accuracy",
        "trend_noise", "movement_change_spearman_rho", "movement_mean", "movement_max",
        "test_reconstruction_mse",
    )
    aggregate_rows = []
    for scope, members in [("all", rows)] + [
        (condition, [row for row in rows if row["condition"] == condition])
        for condition in conditions
    ]:
        for metric in metric_names:
            aggregate_rows.append({"scope": scope, "metric": metric, **confidence_interval([float(row[metric]) for row in members])})
    with (output_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    report = {
        "status": "completed",
        "study": "post_v0.1_seeded_leave_one_bearing_out_stability",
        "conditions": conditions,
        "seeds": seeds,
        "folds_per_condition": 5,
        "run_count": len(rows),
        "protocol": "test one bearing, validate on the next cyclic bearing, train/normalize on the remaining three",
        "parameters": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "latent_dim": args.latent_dim,
            "calibration_steps": args.calibration_steps, "temporal_window": args.temporal_window,
        },
        "aggregate_metrics": aggregate_rows,
        "warnings": [
            "Normalized lifetime is a temporal proxy, not a physical damage label.",
            "Trend and Movement metrics test internal temporal consistency, not maintenance utility.",
            "Runs sharing a fold reuse the same data and are not statistically independent observations.",
        ],
        "duration_seconds": time.monotonic() - started,
        "outputs": ["per_run_metrics.csv", "aggregate_metrics.csv", "study_report.json"],
    }
    with (output_dir / "study_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
