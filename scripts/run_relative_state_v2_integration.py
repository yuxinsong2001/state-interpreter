"""Integrate frozen signed Level, causal Trend and residual-GRU Movement."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from state_interpreter.relative_state_v2 import RelativeStateInterpreterV2
from state_interpreter.residual_gru import ResidualGRUStateInterpreter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(paths):
    models, checkpoints = [], []
    for path in paths:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        model = ResidualGRUStateInterpreter(embedding_dim=checkpoint["embedding_dim"], hidden_dim=checkpoint["hidden_dim"], num_layers=checkpoint["num_layers"])
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        models.append(model); checkpoints.append(checkpoint)
    mean, std = checkpoints[0]["feature_mean"], checkpoints[0]["feature_std"]
    if any(not torch.equal(mean, item["feature_mean"]) or not torch.equal(std, item["feature_std"]) for item in checkpoints[1:]):
        raise ValueError("checkpoint normalization differs")
    return models, mean, std


def ensemble_surprises(sequence, models, mean, std, calibration):
    baseline = sequence[:calibration].mean(0)
    standardized = (sequence - baseline - mean) / std
    with torch.no_grad():
        forecasts = torch.stack([model(standardized[:-1][None]).next_embedding_prediction[0] for model in models]).mean(0)
    return torch.linalg.vector_norm(standardized[1:] - forecasts, dim=1).numpy()


def run_online(sequence, axis, models, mean, std, movement_reference, cfg):
    interpreter = RelativeStateInterpreterV2(axis=axis, residual_models=models, feature_mean=mean, feature_std=std, movement_reference=movement_reference, calibration_steps=cfg["calibration_steps"], temporal_window=cfg["temporal_window"])
    rows = []
    for step, z in enumerate(sequence):
        output = interpreter.update(z)
        if output is not None:
            rows.append({"step_id": step, "level": output.level, "trend": output.trend, "movement": output.movement})
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config_path = Path(args.config).resolve(); cfg = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (ROOT / cfg["input_latents"]).resolve(); checkpoint_dir = (ROOT / cfg["residual_checkpoint_directory"]).resolve()
    axis_path = (ROOT / cfg["signed_axis_result"]).resolve(); output = (ROOT / cfg["output_directory"]).resolve()
    if output.exists(): raise FileExistsError(f"immutable output exists: {output}")
    checkpoint_paths = [checkpoint_dir / f"residual_seed_{seed}_checkpoint.pt" for seed in cfg["random_seeds"]]
    protected = [config_path, input_path, axis_path, *checkpoint_paths]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    axis = np.asarray(json.loads(axis_path.read_text(encoding="utf-8"))["axis"], dtype=float)
    models, mean, std = load_models(checkpoint_paths)
    frame = pd.read_csv(input_path).sort_values(["bearing_id", "step_id"])
    bearings = cfg["training_bearings"] + cfg["evaluation_bearings"]
    frame = frame[frame.bearing_id.isin(bearings)].copy()
    z_cols = sorted([c for c in frame if c.startswith("z_")], key=lambda c: int(c[2:]))
    sequences = {b: torch.tensor(frame[frame.bearing_id == b][z_cols].to_numpy(float), dtype=torch.float32) for b in bearings}
    train_surprises = []
    for bearing in cfg["training_bearings"]:
        values = ensemble_surprises(sequences[bearing], models, mean, std, cfg["calibration_steps"])
        start = cfg["movement_reference_target_start"] - 1
        train_surprises.extend(values[start:].tolist())
    movement_reference = float(np.median(train_surprises))
    if movement_reference <= 0: raise ValueError("invalid movement reference")

    state_rows, summary_rows, integrity = [], [], []
    archived = pd.read_csv(ROOT / "results/2026-09-17_signed_axis_interpreter/per_step_levels.csv")
    for bearing in bearings:
        online = run_online(sequences[bearing], axis, models, mean, std, movement_reference, cfg)
        group = frame[frame.bearing_id == bearing].sort_values("step_id").set_index("step_id")
        for row in online:
            step = row["step_id"]; row.update({"bearing_id": bearing, "split": "training" if bearing in cfg["training_bearings"] else "evaluation", "normalized_lifetime": float(group.loc[step, "normalized_lifetime"])})
            state_rows.append(row)
        current = pd.DataFrame(online)
        reference = archived[archived.bearing_id == bearing].set_index("step_id")
        comparable = current[current.step_id >= cfg["score_start_step"]].set_index("step_id")
        level_error = float(np.max(np.abs(comparable.level - reference.loc[comparable.index, "signed_axis_level"])))
        lifetime = np.array([float(group.loc[s, "normalized_lifetime"]) for s in current.step_id])
        score_mask = current.step_id.to_numpy() >= cfg["score_start_step"]
        late_mask = lifetime >= 0.8
        early_mask = (lifetime < 0.5) & score_mask
        movement = current.movement.to_numpy(float)
        summary_rows.append({
            "bearing_id": bearing, "split": "training" if bearing in cfg["training_bearings"] else "evaluation",
            "state_count": len(current), "level_lifetime_spearman": float(spearmanr(lifetime[score_mask], current.level.to_numpy()[score_mask]).statistic),
            "trend_lifetime_spearman": float(spearmanr(lifetime[score_mask], current.trend.to_numpy()[score_mask]).statistic),
            "movement_lifetime_spearman": float(spearmanr(lifetime[score_mask], movement[score_mask]).statistic),
            "movement_median": float(np.median(movement[score_mask])), "movement_p95": float(np.quantile(movement[score_mask], .95)),
            "late_to_early_movement_ratio": float(np.median(movement[late_mask]) / np.median(movement[early_mask])) if late_mask.any() and early_mask.any() else float("nan"),
            "level_archive_max_abs_error": level_error,
        })
        prefix_length = min(len(sequences[bearing]), cfg["score_start_step"] + 10)
        prefix_rows = run_online(sequences[bearing][:prefix_length], axis, models, mean, std, movement_reference, cfg)
        prefix_error = float(np.max(np.abs(pd.DataFrame(prefix_rows)[["level", "trend", "movement"]].to_numpy() - current.iloc[:len(prefix_rows)][["level", "trend", "movement"]].to_numpy())))
        integrity.append({"bearing_id": bearing, "level_archive_max_abs_error": level_error, "prefix_max_abs_error": prefix_error, "finite": bool(np.isfinite(current[["level", "trend", "movement"]]).all().all()), "movement_nonnegative": bool((current.movement >= 0).all())})

    states = pd.DataFrame(state_rows); summaries = pd.DataFrame(summary_rows); integrity_df = pd.DataFrame(integrity)
    train_median = float(np.median(states[states.bearing_id.isin(cfg["training_bearings"])].movement))
    after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    checks = {
        "archived_level_reproduced": bool((integrity_df.level_archive_max_abs_error <= cfg["integrity_tolerance"]).all()),
        "prefix_causality": bool((integrity_df.prefix_max_abs_error <= cfg["integrity_tolerance"]).all()),
        "all_finite": bool(integrity_df.finite.all()), "movement_nonnegative": bool(integrity_df.movement_nonnegative.all()),
        "training_movement_median_is_one": bool(abs(train_median - 1.0) <= cfg["integrity_tolerance"]),
        "protected_hashes_unchanged": before == after,
    }
    output.mkdir(parents=True)
    states.to_csv(output / "relative_state_trajectories.csv", index=False); summaries.to_csv(output / "summary_by_bearing.csv", index=False); integrity_df.to_csv(output / "integrity_checks.csv", index=False)
    fig, axes = plt.subplots(len(cfg["evaluation_bearings"]), 3, figsize=(14, 7), sharex="row")
    for row_index, bearing in enumerate(cfg["evaluation_bearings"]):
        part = states[states.bearing_id == bearing]
        for ax, column, title in zip(axes[row_index], ["level", "trend", "movement"], ["Signed Level", "Causal Trend", "GRU Movement surprise"]):
            ax.plot(part.normalized_lifetime, part[column], lw=1.5); ax.set_title(f"{bearing}: {title}"); ax.grid(alpha=.25); ax.set_xlabel("Normalized lifetime")
    fig.tight_layout(); fig.savefig(output / "relative_state_v2.png", dpi=180); plt.close(fig)
    report = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256(config_path), "input_sha256": sha256(input_path), "movement_reference": movement_reference, "training_movement_median": train_median, "artifact_hashes": before, "integrity_checks": checks, "all_integrity_checks_pass": bool(all(checks.values())), "interpretation": "Relative signals only; no absolute damage, RUL or physical health claim."}
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summaries.to_string(index=False)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
