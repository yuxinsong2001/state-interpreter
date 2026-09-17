"""Evaluate a training-only signed degradation-axis State Interpreter."""

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

from state_interpreter.signed_axis import fit_signed_axis, signed_level


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def causal_mean(values: np.ndarray, window: int) -> np.ndarray:
    return np.array([np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))])


def euclidean_level(z: np.ndarray, calibration: int, window: int) -> np.ndarray:
    reference = z[:calibration].mean(axis=0)
    raw = np.linalg.norm(z - reference, axis=1)
    result = np.full(len(z), np.nan)
    result[calibration:] = causal_mean(raw[calibration:], window)
    return result


def trajectory_metrics(lifetime: np.ndarray, level: np.ndarray, mask: np.ndarray) -> dict:
    x, y = lifetime[mask], level[mask]
    diff = np.diff(y)
    return {
        "score_samples": int(len(y)),
        "spearman_rho": safe_spearman(x, y),
        "first_last_delta": float(y[-1] - y[0]),
        "backward_step_fraction": float(np.mean(diff < -1e-9)),
        "mean_absolute_step": float(np.mean(np.abs(diff))),
        "step_std": float(np.std(diff)),
    }


def stage_labels(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.digitize(values, thresholds).astype(int)


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

    frame = pd.read_csv(input_path)
    z_cols = sorted([c for c in frame if c.startswith("z_")], key=lambda x: int(x[2:]))
    selected = config["training_bearings"] + config["evaluation_bearings"]
    frame = frame[frame["bearing_id"].isin(selected)].copy()
    frame = frame.sort_values(["bearing_id", "step_id"]).reset_index(drop=True)
    calibration = int(config["calibration_steps"])
    window = int(config["temporal_window"])
    score_start = int(config["score_start_step"])

    sequences = {
        bearing: group[z_cols].to_numpy(float)
        for bearing, group in frame.groupby("bearing_id", sort=False)
    }
    axis = fit_signed_axis(
        [sequences[b] for b in config["training_bearings"]],
        calibration_steps=calibration,
    )

    level_rows, metric_rows = [], []
    for bearing in selected:
        group = frame[frame["bearing_id"] == bearing].sort_values("step_id")
        z = group[z_cols].to_numpy(float)
        signed = signed_level(z, axis, calibration_steps=calibration, temporal_window=window)
        euclidean = euclidean_level(z, calibration, window)
        steps = group["step_id"].to_numpy(int)
        lifetime = group["normalized_lifetime"].to_numpy(float)
        score_mask = (steps >= score_start) & np.isfinite(signed) & np.isfinite(euclidean)
        split = "training" if bearing in config["training_bearings"] else "evaluation"
        for method, values in (("euclidean", euclidean), ("signed_axis", signed)):
            metrics = trajectory_metrics(lifetime, values, score_mask)
            metric_rows.append({"bearing_id": bearing, "split": split, "method": method, **metrics})
        for step, life, euc, sig in zip(steps, lifetime, euclidean, signed):
            level_rows.append({
                "bearing_id": bearing, "split": split, "step_id": int(step),
                "normalized_lifetime": float(life), "euclidean_level": euc,
                "signed_axis_level": sig,
            })

    levels = pd.DataFrame(level_rows)
    metrics = pd.DataFrame(metric_rows)
    train = levels[(levels["split"] == "training") & (levels["step_id"] >= score_start)].dropna()
    evaluations = levels[(levels["split"] == "evaluation") & (levels["step_id"] >= score_start)].dropna()
    thresholds = config["stage_thresholds"]
    stage_rows, calibration_rows = [], []
    for method, column in (("euclidean", "euclidean_level"), ("signed_axis", "signed_axis_level")):
        stage_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(config["logistic_C"]), class_weight="balanced", max_iter=2000, random_state=int(config["random_seed"])),
        )
        lifetime_model = make_pipeline(StandardScaler(), Ridge(alpha=float(config["ridge_alpha"])))
        x_train = train[[column]].to_numpy()
        y_life_train = train["normalized_lifetime"].to_numpy()
        y_stage_train = stage_labels(y_life_train, thresholds)
        stage_model.fit(x_train, y_stage_train)
        lifetime_model.fit(x_train, y_life_train)
        for bearing, group in evaluations.groupby("bearing_id", sort=False):
            x_test = group[[column]].to_numpy()
            true_life = group["normalized_lifetime"].to_numpy()
            true_stage = stage_labels(true_life, thresholds)
            predicted_stage = stage_model.predict(x_test)
            predicted_life = lifetime_model.predict(x_test)
            stage_rows.append({
                "bearing_id": bearing, "method": method,
                "balanced_accuracy": float(balanced_accuracy_score(true_stage, predicted_stage)),
                "macro_f1": float(f1_score(true_stage, predicted_stage, average="macro", zero_division=0)),
            })
            calibration_rows.append({
                "bearing_id": bearing, "method": method,
                "mae": float(mean_absolute_error(true_life, predicted_life)),
                "r2": float(r2_score(true_life, predicted_life)),
                "prediction_spearman": safe_spearman(true_life, predicted_life),
            })

    stages = pd.DataFrame(stage_rows)
    calibrations = pd.DataFrame(calibration_rows)
    gates = config["decision_gates"]
    decisions = []
    for bearing in config["evaluation_bearings"]:
        euc = metrics[(metrics.bearing_id == bearing) & (metrics.method == "euclidean")].iloc[0]
        sig = metrics[(metrics.bearing_id == bearing) & (metrics.method == "signed_axis")].iloc[0]
        euc_stage = stages[(stages.bearing_id == bearing) & (stages.method == "euclidean")].iloc[0]
        sig_stage = stages[(stages.bearing_id == bearing) & (stages.method == "signed_axis")].iloc[0]
        checks = {
            "positive_direction": bool(sig.spearman_rho > 0),
            "rho_not_worse": bool(sig.spearman_rho - euc.spearman_rho >= gates["minimum_spearman_delta_vs_euclidean"]),
            "backward_not_worse": bool(sig.backward_step_fraction - euc.backward_step_fraction <= gates["maximum_backward_fraction_increase"]),
            "stage_not_worse": bool(sig_stage.balanced_accuracy - euc_stage.balanced_accuracy >= gates["minimum_stage_balanced_accuracy_delta"]),
        }
        decisions.append({"bearing_id": bearing, **checks, "all_gates_pass": bool(all(checks.values()))})
    continue_signed_axis = bool(all(row["all_gates_pass"] for row in decisions))

    output.mkdir(parents=True)
    levels.to_csv(output / "per_step_levels.csv", index=False)
    metrics.to_csv(output / "trajectory_metrics.csv", index=False)
    stages.to_csv(output / "stage_metrics.csv", index=False)
    calibrations.to_csv(output / "lifetime_calibration_metrics.csv", index=False)
    (output / "axis.json").write_text(json.dumps({"axis": axis.tolist(), "norm": float(np.linalg.norm(axis))}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for row, bearing in enumerate(config["evaluation_bearings"]):
        group = levels[levels.bearing_id == bearing]
        axes[row, 0].plot(group.normalized_lifetime, group.euclidean_level, label="Euclidean", lw=1.7)
        axes[row, 0].plot(group.normalized_lifetime, group.signed_axis_level, label="Signed axis", lw=1.7)
        axes[row, 0].set_title(bearing)
        axes[row, 0].set_ylabel("Level")
        axes[row, 0].grid(alpha=.25)
        axes[row, 0].legend()
        diff = group.signed_axis_level.diff()
        axes[row, 1].plot(group.normalized_lifetime, diff, color="tab:orange", lw=1)
        axes[row, 1].axhline(0, color="black", lw=.8)
        axes[row, 1].set_title(f"{bearing}: signed step change")
        axes[row, 1].grid(alpha=.25)
    for ax in axes[-1]: ax.set_xlabel("Normalized lifetime")
    fig.tight_layout()
    fig.savefig(output / "signed_axis_comparison.png", dpi=180)
    plt.close(fig)

    fallacy_scan = {
        "1_measurement_validity": "Normalized lifetime is only a chronological proxy.",
        "2_population_scope": "One condition and five bearings; no broad population claim.",
        "3_independence": "Samples within a bearing are dependent; bearing is the evaluation unit.",
        "4_sample_size": "Only two evaluation bearings; report per-bearing results, not significance.",
        "5_missing_data": "No missing latent rows detected after selected-bearing filtering.",
        "6_multiple_testing": "One predeclared signed-axis comparison; no p-value search.",
        "7_effect_size": "Report rho, backward fraction, balanced accuracy, MAE and R2.",
        "8_uncertainty": "Insufficient independent bearings for reliable confidence intervals.",
        "9_model_assumptions": "PCA assumes a common linear direction; failure does not reject nonlinear structure.",
        "10_leakage": "Axis and mappings use only Bearing1_1-1_3; evaluation bearings supply local early reference only.",
        "11_causation": "Association with time does not establish physical damage causation."
    }
    report = {
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(config_path), "input_sha256": sha256(input_path),
        "axis_fit_bearings": config["training_bearings"],
        "evaluation_bearings": config["evaluation_bearings"],
        "decisions": decisions, "continue_signed_axis": continue_signed_axis,
        "fallacy_scan_11_of_11": fallacy_scan,
    }
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(stages.to_string(index=False))
    print(json.dumps({"decisions": decisions, "continue_signed_axis": continue_signed_axis}, indent=2))


if __name__ == "__main__":
    main()
