"""Select an early-window scale on training bearings, then evaluate it frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
repository_src = repository_root / "src"
if str(repository_src) not in sys.path:
    sys.path.insert(0, str(repository_src))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from state_interpreter.early_scale import apply_scale, early_scale
from state_interpreter.signed_axis import fit_signed_axis, signed_level


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stages(y: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.digitize(y, thresholds).astype(int)


def rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic) if len(x) > 1 and np.ptp(y) > 0 else float("nan")


def fit_models(x: np.ndarray, lifetime: np.ndarray, config: dict):
    stage_model = make_pipeline(StandardScaler(), LogisticRegression(
        C=float(config["logistic_C"]), class_weight="balanced", max_iter=2000,
        random_state=int(config["random_seed"])))
    life_model = make_pipeline(StandardScaler(), Ridge(alpha=float(config["ridge_alpha"])))
    stage_model.fit(x[:, None], stages(lifetime, config["stage_thresholds"]))
    life_model.fit(x[:, None], lifetime)
    return stage_model, life_model


def make_levels(sequences, bearings, axis, method, config, floor_source):
    calibration = int(config["calibration_steps"])
    window = int(config["temporal_window"])
    raw_scales = {b: early_scale(sequences[b], axis, calibration_steps=calibration, method=method) for b in bearings + floor_source}
    training_scales = [raw_scales[b] for b in floor_source]
    median = float(np.median(training_scales))
    floor = 1.0 if method == "none" else float(config["scale_floor_fraction_of_training_median"]) * median
    if floor <= 0:
        raise ValueError("training-derived scale floor is not positive")
    levels = {}
    for b in bearings:
        raw = signed_level(sequences[b], axis, calibration_steps=calibration, temporal_window=window)
        levels[b] = apply_scale(raw, raw_scales[b], floor=floor)
    return levels, {b: raw_scales[b] for b in bearings}, floor


def vectors(frame, bearings, z_cols):
    return {b: frame[frame.bearing_id == b].sort_values("step_id")[z_cols].to_numpy(float) for b in bearings}


def selected_rows(frame, bearing, level, score_start):
    group = frame[frame.bearing_id == bearing].sort_values("step_id")
    mask = (group.step_id.to_numpy(int) >= score_start) & np.isfinite(level)
    return level[mask], group.normalized_lifetime.to_numpy(float)[mask]


def evaluate_fold(frame, sequences, train_bearings, held, method, config, z_cols):
    axis = fit_signed_axis([sequences[b] for b in train_bearings], calibration_steps=int(config["calibration_steps"]))
    levels, scales, floor = make_levels(sequences, train_bearings + [held], axis, method, config, train_bearings)
    train_x, train_y = zip(*(selected_rows(frame, b, levels[b], int(config["score_start_step"])) for b in train_bearings))
    x_train, y_train = np.concatenate(train_x), np.concatenate(train_y)
    stage_model, life_model = fit_models(x_train, y_train, config)
    x_test, y_test = selected_rows(frame, held, levels[held], int(config["score_start_step"]))
    predicted_stage = stage_model.predict(x_test[:, None])
    predicted_life = life_model.predict(x_test[:, None])
    return {
        "held_bearing": held, "method": method,
        "balanced_accuracy": float(balanced_accuracy_score(stages(y_test, config["stage_thresholds"]), predicted_stage)),
        "macro_f1": float(f1_score(stages(y_test, config["stage_thresholds"]), predicted_stage, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_test, predicted_life)), "r2": float(r2_score(y_test, predicted_life)),
        "level_spearman": rho(y_test, x_test), "held_scale": float(scales[held]), "training_floor": floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (repository_root / config["input_latents"]).resolve()
    output = (repository_root / config["output_directory"]).resolve()
    if output.exists():
        raise FileExistsError(f"immutable output directory already exists: {output}")
    frame = pd.read_csv(input_path).sort_values(["bearing_id", "step_id"])
    all_bearings = config["training_bearings"] + config["evaluation_bearings"]
    frame = frame[frame.bearing_id.isin(all_bearings)].copy()
    z_cols = sorted([c for c in frame if c.startswith("z_")], key=lambda c: int(c[2:]))
    sequences = vectors(frame, all_bearings, z_cols)

    cv_rows = []
    for method in config["candidate_methods"]:
        for held in config["training_bearings"]:
            train = [b for b in config["training_bearings"] if b != held]
            cv_rows.append(evaluate_fold(frame, sequences, train, held, method, config, z_cols))
    cv = pd.DataFrame(cv_rows)
    summary = cv.groupby("method", sort=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_macro_f1=("macro_f1", "mean"),
        mean_mae=("mae", "mean"), mean_r2=("r2", "mean"), mean_level_spearman=("level_spearman", "mean")
    ).reset_index()
    order = {m: i for i, m in enumerate(config["candidate_methods"])}
    summary["candidate_order"] = summary.method.map(order)
    selected_method = summary.sort_values(
        ["mean_balanced_accuracy", "mean_mae", "candidate_order"], ascending=[False, True, True]
    ).iloc[0].method

    train_bearings = config["training_bearings"]
    eval_bearings = config["evaluation_bearings"]
    axis = fit_signed_axis([sequences[b] for b in train_bearings], calibration_steps=int(config["calibration_steps"]))
    evaluated_methods = ["none"] if selected_method == "none" else ["none", selected_method]
    final_rows, per_step_rows, scale_rows = [], [], []
    for method in evaluated_methods:
        levels, scales, floor = make_levels(sequences, train_bearings + eval_bearings, axis, method, config, train_bearings)
        train_xy = [selected_rows(frame, b, levels[b], int(config["score_start_step"])) for b in train_bearings]
        x_train = np.concatenate([x for x, _ in train_xy]); y_train = np.concatenate([y for _, y in train_xy])
        stage_model, life_model = fit_models(x_train, y_train, config)
        for bearing in train_bearings + eval_bearings:
            scale_rows.append({"bearing_id": bearing, "method": method, "raw_early_scale": scales[bearing], "training_floor": floor, "used_scale": max(scales[bearing], floor)})
        for bearing in eval_bearings:
            x_test, y_test = selected_rows(frame, bearing, levels[bearing], int(config["score_start_step"]))
            stage_pred = stage_model.predict(x_test[:, None]); life_pred = life_model.predict(x_test[:, None])
            final_rows.append({
                "bearing_id": bearing, "method": method, "score_samples": len(x_test),
                "balanced_accuracy": float(balanced_accuracy_score(stages(y_test, config["stage_thresholds"]), stage_pred)),
                "macro_f1": float(f1_score(stages(y_test, config["stage_thresholds"]), stage_pred, average="macro", zero_division=0)),
                "mae": float(mean_absolute_error(y_test, life_pred)), "r2": float(r2_score(y_test, life_pred)),
                "level_spearman": rho(y_test, x_test), "prediction_spearman": rho(y_test, life_pred),
            })
            group = frame[frame.bearing_id == bearing].sort_values("step_id")
            for step, life, value in zip(group.step_id, group.normalized_lifetime, levels[bearing]):
                per_step_rows.append({"bearing_id": bearing, "method": method, "step_id": int(step), "normalized_lifetime": float(life), "level": value})

    final = pd.DataFrame(final_rows); scales_df = pd.DataFrame(scale_rows); per_step = pd.DataFrame(per_step_rows)
    baseline = final[final.method == "none"].set_index("bearing_id")
    chosen = final[final.method == selected_method].set_index("bearing_id")
    gates = config["evaluation_gates"]
    if selected_method == "none":
        decisions = {"selected_is_baseline": True, "stage_gate": False, "mae_gate": False, "single_bearing_mae_gate": False, "adopt_scale": False}
    else:
        stage_delta = float(chosen.balanced_accuracy.mean() - baseline.balanced_accuracy.mean())
        mae_improvement = float(baseline.mae.mean() - chosen.mae.mean())
        worst_increase = float(((chosen.mae - baseline.mae) / baseline.mae).max())
        decisions = {
            "selected_is_baseline": False,
            "stage_gate": bool(stage_delta > gates["minimum_mean_stage_balanced_accuracy_delta_vs_none"]),
            "mae_gate": bool(mae_improvement > gates["minimum_mean_mae_improvement_vs_none"]),
            "single_bearing_mae_gate": bool(worst_increase <= gates["maximum_single_bearing_mae_increase_fraction"]),
            "mean_stage_delta": stage_delta, "mean_mae_improvement": mae_improvement,
            "worst_single_bearing_mae_increase_fraction": worst_increase,
        }
        decisions["adopt_scale"] = bool(decisions["stage_gate"] and decisions["mae_gate"] and decisions["single_bearing_mae_gate"])

    output.mkdir(parents=True)
    cv.to_csv(output / "training_lobo_folds.csv", index=False)
    summary.to_csv(output / "training_lobo_summary.csv", index=False)
    final.to_csv(output / "evaluation_metrics.csv", index=False)
    scales_df.to_csv(output / "early_scales.csv", index=False)
    per_step.to_csv(output / "evaluation_per_step.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, bearing in zip(axes, eval_bearings):
        for method in evaluated_methods:
            subset = per_step[(per_step.bearing_id == bearing) & (per_step.method == method)]
            ax.plot(subset.normalized_lifetime, subset.level, label=method, lw=1.7)
        ax.set_title(bearing); ax.set_xlabel("Normalized lifetime"); ax.set_ylabel("Calibrated Level"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output / "early_scale_comparison.png", dpi=180); plt.close(fig)
    fallacy_scan = {
        "1_simpson": "Per-bearing and mean results retained.", "2_ecological": "Inference limited to bearing trajectories.",
        "3_berkson": "Selected run-to-failure dataset may not represent field population.", "4_collider": "No covariate adjustment.",
        "5_base_rate": "Balanced accuracy reported; stage prevalence remains dataset-specific.", "6_regression_to_mean": "Early reference is protocol-defined, not selected for extreme values.",
        "7_survivorship": "Complete archived sequences used; field censoring not represented.", "8_look_elsewhere": "Three candidates and selection rule preregistered.",
        "9_forking_paths": "Configuration fixed before evaluation and outputs immutable.", "10_correlation_causation": "Time association is not physical damage causation.",
        "11_reverse_causality": "Chronology is known, but the latent mechanism is not causal evidence."
    }
    report = {
        "experiment_id": config["experiment_id"], "config_sha256": sha256(config_path), "input_sha256": sha256(input_path),
        "selected_method_from_training_lobo": selected_method, "evaluated_methods": evaluated_methods,
        "decisions": decisions, "fallacy_scan_11_of_11": fallacy_scan,
    }
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(final.to_string(index=False)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
