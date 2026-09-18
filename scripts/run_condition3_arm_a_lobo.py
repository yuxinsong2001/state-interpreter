"""Condition 3 Arm A: LOBO interpreter adaptation with a frozen encoder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from state_interpreter.adapters import XJTUSYDatasetAdapter
from state_interpreter.data import materialize_stft_data
from state_interpreter.residual_gru import ResidualGRUStateInterpreter
from state_interpreter.signed_axis import fit_signed_axis, signed_level
from run_direct_generalization_v2 import load_frozen_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_residuals(paths: list[Path]):
    models, mean, std = [], None, None
    for path in paths:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        model = ResidualGRUStateInterpreter(
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
        if mean is None:
            mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
        elif not torch.equal(mean, checkpoint["feature_mean"]) or not torch.equal(std, checkpoint["feature_std"]):
            raise ValueError("Residual-GRU normalization mismatch")
    return models, mean, std


def raw_movement(sequence: np.ndarray, models, mean: torch.Tensor, std: torch.Tensor, calibration: int) -> np.ndarray:
    values = torch.tensor(sequence, dtype=torch.float32)
    baseline = values[:calibration].mean(0)
    standardized = (values - baseline - mean) / std
    output = np.full(len(sequence), np.nan)
    with torch.inference_mode():
        for index in range(calibration, len(sequence)):
            forecasts = [m(standardized[:index][None]).next_embedding_prediction[0, -1] for m in models]
            output[index] = float(torch.linalg.vector_norm(standardized[index] - torch.stack(forecasts).mean(0)))
    return output


def metrics(lifetime: np.ndarray, level: np.ndarray, movement: np.ndarray, steps: np.ndarray, threshold: float, score_start: int):
    mask = (steps >= score_start) & np.isfinite(level) & np.isfinite(movement)
    x, y, m = lifetime[mask], level[mask], movement[mask]
    differences = np.diff(y)
    return {
        "score_samples": int(mask.sum()),
        "level_spearman": float(spearmanr(x, y).statistic),
        "level_first_last_delta": float(y[-1] - y[0]),
        "backward_step_fraction": float(np.mean(differences < -1e-9)),
        "movement_median": float(np.median(m)),
        "movement_event_fraction": float(np.mean(m >= threshold)),
        "movement_max": float(np.max(m)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != "EXECUTE_CONDITION3_ARM_A_LOBO_ONCE":
        raise PermissionError("Exact Arm A confirmation token required")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_development_only":
        raise ValueError("Invalid preregistration status")
    if config["bearings"] != ["Bearing3_1", "Bearing3_2", "Bearing3_3"] or config["protected_bearings"] != ["Bearing3_4", "Bearing3_5"]:
        raise ValueError("Development/protected bearing boundary changed")
    expected = {row["test"]: tuple(row["train"]) for row in config["folds"]}
    if expected != {"Bearing3_1": ("Bearing3_2", "Bearing3_3"), "Bearing3_2": ("Bearing3_1", "Bearing3_3"), "Bearing3_3": ("Bearing3_1", "Bearing3_2")}:
        raise ValueError("LOBO folds changed")
    output = ROOT / config["output_directory"]
    if output.exists():
        raise FileExistsError(f"Immutable output exists: {output}")
    if not (ROOT / "records/xjtu_condition3_frozen_v2/development_completed.json").is_file():
        raise FileNotFoundError("Frozen development diagnostic must complete first")

    _, autoencoder, preprocessor, standardizer = load_frozen_model(ROOT / config["frozen_autoencoder_checkpoint"])
    models, feature_mean, feature_std = load_residuals([ROOT / p for p in config["frozen_residual_checkpoints"]])
    sequences, metadata, reconstruction = {}, {}, {}
    started = time.time()
    for bearing in config["bearings"]:
        adapter = XJTUSYDatasetAdapter(Path(args.root), conditions=[config["condition"]], bearings=[bearing])
        data = materialize_stft_data(adapter, preprocessor)
        inputs = standardizer.transform(data.tensors)
        chunks, errors = [], []
        with torch.inference_mode():
            for begin in range(0, len(inputs), 32):
                result = autoencoder(inputs[begin:begin + 32])
                chunks.append(result.z.cpu())
                errors.append((result.reconstruction - inputs[begin:begin + 32]).square().mean((1, 2, 3)).cpu())
        sequences[bearing] = torch.cat(chunks).numpy()
        reconstruction[bearing] = torch.cat(errors).numpy()
        steps = np.asarray(data.step_ids, dtype=int)
        metadata[bearing] = {"steps": steps, "lifetime": steps / max(steps)}

    movement_raw = {
        bearing: raw_movement(sequence, models, feature_mean, feature_std, int(config["calibration_steps"]))
        for bearing, sequence in sequences.items()
    }
    baseline = pd.read_csv(ROOT / config["source_frozen_result"]).set_index("bearing_id")
    rows, state_rows, fold_artifacts = [], [], {}
    for fold in config["folds"]:
        test, train = fold["test"], fold["train"]
        axis = fit_signed_axis([sequences[b] for b in train], calibration_steps=int(config["calibration_steps"]))
        training_movement = np.concatenate([
            movement_raw[b][metadata[b]["steps"] >= int(config["score_start_step"])] for b in train
        ])
        training_movement = training_movement[np.isfinite(training_movement)]
        reference = float(np.median(training_movement))
        threshold = float(np.quantile(training_movement / reference, float(config["movement_quantile"])))
        level = signed_level(sequences[test], axis, calibration_steps=int(config["calibration_steps"]), temporal_window=int(config["temporal_window"]))
        movement = movement_raw[test] / reference
        result = metrics(metadata[test]["lifetime"], level, movement, metadata[test]["steps"], threshold, int(config["score_start_step"]))
        old = baseline.loc[test]
        rows.append({
            "test_bearing": test,
            "train_bearings": "+".join(train),
            **result,
            "baseline_level_spearman": float(old.level_spearman),
            "spearman_delta_vs_baseline": result["level_spearman"] - float(old.level_spearman),
            "baseline_movement_event_fraction": float(old.movement_event_fraction),
            "movement_event_fraction_delta_vs_baseline": result["movement_event_fraction"] - float(old.movement_event_fraction),
            "movement_reference_raw": reference,
            "movement_threshold_normalized": threshold,
            "reconstruction_mse_mean": float(np.mean(reconstruction[test])),
        })
        fold_artifacts[test] = {"train": train, "axis": axis.tolist(), "movement_reference_raw": reference, "movement_threshold_normalized": threshold}
        for step, life, lev, mov in zip(metadata[test]["steps"], metadata[test]["lifetime"], level, movement):
            state_rows.append({"bearing_id": test, "step_id": int(step), "normalized_lifetime": float(life), "level": lev, "movement": mov})

    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "lobo_summary.csv", index=False)
    pd.DataFrame(state_rows).to_csv(output / "lobo_state_trajectories.csv", index=False)
    (output / "fold_artifacts.json").write_text(json.dumps(fold_artifacts, indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    decision = {
        "majority_level_rho_improved": bool((summary.spearman_delta_vs_baseline > 0).sum() >= 2),
        "all_level_rho_positive": bool((summary.level_spearman > 0).all()),
        "proceed_to_encoder_adaptation": bool(not (summary.level_spearman > 0).all()),
    }
    report = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "config_sha256": sha256(config_path),
        "protected_bearings_not_read": config["protected_bearings"],
        "elapsed_seconds": time.time() - started,
        "decision": decision,
        "fallacy_scan_11_of_11": {
            "simpsons_paradox": "Per-bearing LOBO results are primary; no pooled trajectory correlation.",
            "ecological_fallacy": "Bearing is the evaluation unit; time samples are dependent.",
            "berkson_bias": "Laboratory run-to-failure bearings are selected trajectories.",
            "collider_bias": "No conditioned causal comparison is made.",
            "base_rate_neglect": "Movement event fraction is reported per held-out bearing.",
            "regression_to_mean": "No intervention or before-after causal claim.",
            "survivorship_bias": "Complete trajectories do not represent field censoring.",
            "look_elsewhere_effect": "Three folds and metrics were fixed before execution.",
            "researcher_degrees_of_freedom": "Encoder, folds, calibration length and quantile were preregistered.",
            "correlation_causation": "Time ordering is not physical damage causation.",
            "reverse_causality": "Chronology is preserved; no causal mechanism is claimed."
        }
    }
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    states = pd.DataFrame(state_rows)
    for bearing, group in states.groupby("bearing_id", sort=False):
        axes[0].plot(group.normalized_lifetime, group.level, label=bearing)
        axes[1].plot(group.normalized_lifetime, group.movement, label=bearing)
    axes[0].set_ylabel("LOBO adapted Level")
    axes[1].set_ylabel("LOBO normalized Movement")
    axes[1].set_xlabel("Normalized lifetime")
    for axis in axes:
        axis.grid(alpha=.3); axis.legend()
    fig.tight_layout(); fig.savefig(output / "lobo_trajectories.png", dpi=180); plt.close(fig)
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
