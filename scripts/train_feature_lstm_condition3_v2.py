"""Run the preregistered 3-fold x 3-seed Feature LSTM development study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.encoders.feature_lstm import FeatureLSTM
from state_interpreter.feature_lstm_experiment import (
    backward_step_fraction,
    prepare_bearing_windows,
    set_deterministic_seed,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config["status"] != "preregistered_training_v2_signed_log1p":
        raise ValueError("invalid training v2 status")
    if args.confirm != config["execution_token"]:
        raise PermissionError("exact Feature LSTM v2 training token required")
    expected = ["Bearing3_1", "Bearing3_2", "Bearing3_3"]
    if config["development_bearings"] != expected:
        raise ValueError("development split changed")
    if config["protected_validation"] != ["Bearing3_4"]:
        raise ValueError("validation protection changed")
    if config["protected_blind"] != ["Bearing3_5"]:
        raise ValueError("blind protection changed")
    if config["input"]["scaling_mode"] != "signed_log1p":
        raise ValueError("stable scaling amendment not applied")
    for artifact in config["frozen_artifacts"]:
        if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"frozen artifact mismatch: {artifact['path']}")

    final_output = (ROOT / config["output_directory"]).resolve()
    staging = final_output.with_name(final_output.name + ".staging")
    if final_output.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite training output or staging data")

    cache = (ROOT / config["input"]["cache_directory"]).resolve()
    prepared = {}
    for entry in config["input"]["cache_entries"]:
        bearing = entry["bearing_id"]
        if bearing not in expected:
            raise ValueError("cache entry outside development scope")
        path = cache / f"{bearing}.npz"
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"cache hash mismatch: {bearing}")
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
        prepared[bearing] = prepare_bearing_windows(
            features,
            bearing_id=bearing,
            calibration_steps=config["input"]["calibration_steps"],
            window_size=config["input"]["window_size"],
            scaling_mode=config["input"]["scaling_mode"],
        )

    staging.mkdir(parents=True)
    started = time.time()
    metric_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    model_config = config["model"]
    training = config["training"]
    for seed in training["seeds"]:
        for fold in config["outer_lobo_folds"]:
            test_bearing = fold["test"]
            train_bearings = fold["train"]
            set_deterministic_seed(seed)
            train_windows = torch.cat(
                [prepared[bearing].windows for bearing in train_bearings]
            )
            train_targets = torch.cat(
                [prepared[bearing].remaining_useful_life for bearing in train_bearings]
            )
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                TensorDataset(train_windows, train_targets),
                batch_size=training["batch_size"],
                shuffle=True,
                generator=generator,
            )
            model = FeatureLSTM(
                n_features=model_config["n_features"],
                hidden_size=model_config["hidden_size"],
                dense_size=model_config["dense_size"],
                dropout=model_config["dropout"],
                bidirectional=model_config["bidirectional"],
            )
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_count != model_config["parameter_count"]:
                raise ValueError("Feature LSTM parameter count changed")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training["learning_rate"],
                weight_decay=training["weight_decay"],
            )
            loss_function = nn.HuberLoss(delta=training["huber_delta"])
            for epoch in range(1, training["fixed_epochs"] + 1):
                model.train()
                total_loss = 0.0
                count = 0
                for windows, targets in loader:
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(windows).score.squeeze(1)
                    loss = loss_function(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach()) * windows.shape[0]
                    count += windows.shape[0]
                mean_loss = total_loss / count
                history_rows.append(
                    {
                        "seed": seed,
                        "test_bearing": test_bearing,
                        "epoch": epoch,
                        "training_huber_loss": mean_loss,
                    }
                )
                if epoch in (1, 10, 20, 30, 40, 50):
                    print(
                        f"seed={seed} test={test_bearing} epoch={epoch}/50 "
                        f"loss={mean_loss:.6f}",
                        flush=True,
                    )

            checkpoint = staging / f"seed_{seed}_{test_bearing}_final.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "test_bearing": test_bearing,
                    "train_bearings": train_bearings,
                    "epoch": training["fixed_epochs"],
                    "scaling_mode": config["input"]["scaling_mode"],
                },
                checkpoint,
            )
            model.eval()
            test = prepared[test_bearing]
            prediction_batches = []
            with torch.inference_mode():
                for start in range(0, test.windows.shape[0], training["batch_size"]):
                    prediction_batches.append(
                        model(test.windows[start : start + training["batch_size"]])
                        .score.squeeze(1)
                        .cpu()
                    )
            predicted_rul = torch.cat(prediction_batches).numpy()
            progress = 1.0 - np.clip(predicted_rul, 0.0, 1.0)
            lifetime = test.normalized_lifetime.numpy()
            rho = float(spearmanr(lifetime, progress).statistic)
            backward = backward_step_fraction(progress)
            metric_rows.append(
                {
                    "seed": seed,
                    "test_bearing": test_bearing,
                    "train_bearings": "+".join(train_bearings),
                    "window_count": int(test.windows.shape[0]),
                    "spearman_progress": rho,
                    "backward_step_fraction": backward,
                    "final_training_loss": history_rows[-1]["training_huber_loss"],
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
            for step, life, raw_rul, state in zip(
                test.end_step_ids.tolist(), lifetime, predicted_rul, progress
            ):
                trajectory_rows.append(
                    {
                        "seed": seed,
                        "bearing_id": test_bearing,
                        "step_id": step,
                        "normalized_lifetime": float(life),
                        "predicted_rul": float(raw_rul),
                        "health_progress": float(state),
                    }
                )

    summary_rows: list[dict[str, object]] = []
    for bearing in expected:
        rows = [row for row in metric_rows if row["test_bearing"] == bearing]
        rhos = np.asarray([row["spearman_progress"] for row in rows], dtype=float)
        backward = np.asarray(
            [row["backward_step_fraction"] for row in rows], dtype=float
        )
        summary_rows.append(
            {
                "bearing_id": bearing,
                "mean_spearman": float(rhos.mean()),
                "std_spearman": float(rhos.std(ddof=0)),
                "positive_seed_count": int(np.count_nonzero(rhos > 0.0)),
                "mean_backward_step_fraction": float(backward.mean()),
            }
        )
    gate_config = config["evaluation"]["gate"]
    mean_rhos = [float(row["mean_spearman"]) for row in summary_rows]
    gate = {
        "all_three_mean_spearman_positive": all(value > 0.0 for value in mean_rhos),
        "at_least_two_mean_spearman_at_least_0_5": sum(
            value >= gate_config["at_least_two_mean_spearman_at_least"]
            for value in mean_rhos
        )
        >= 2,
        "minimum_positive_seeds_per_bearing": all(
            int(row["positive_seed_count"])
            >= gate_config["minimum_positive_seeds_per_bearing"]
            for row in summary_rows
        ),
    }
    gate["passed"] = all(gate.values())
    write_csv(staging / "per_seed_bearing_metrics.csv", metric_rows)
    write_csv(staging / "summary_by_bearing.csv", summary_rows)
    write_csv(staging / "training_history.csv", history_rows)
    write_csv(staging / "trajectories.csv", trajectory_rows)
    report = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "config_sha256": sha256(config_path),
        "summary": summary_rows,
        "validation_gate": gate,
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "fixed_final_epoch_used": True,
        "outer_test_used_for_checkpoint_selection": False,
    }
    (staging / "experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    staging.rename(final_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
