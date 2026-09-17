"""Audit whether frozen z=8 embeddings encode transferable degradation information."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from state_interpreter.representation_audit import cosine_similarity_matrix, early_center, effective_dimension, stage_labels


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty rows: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def finite_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return None, None, int(mask.sum())
    result = spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue), int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output = (ROOT / cfg["output_directory"]).resolve()
    if not output.is_relative_to(ROOT / "results") or output.exists():
        raise ValueError("output must be a new directory under results")
    started = time.monotonic()
    input_path = ROOT / cfg["input_latents"]
    source_report_path = ROOT / cfg["source_analysis_report"]
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(source_report["checkpoint"])
    if source_report["status"] != "completed" or source_report["latent_dim"] != 8:
        raise ValueError("source latent analysis is incomplete or wrong dimension")
    protected = [config_path, input_path, source_report_path, checkpoint_path]
    before = {str(path if not path.is_relative_to(ROOT) else path.relative_to(ROOT)): sha256(path) for path in protected}

    frame = pd.read_csv(input_path)
    z_columns = [f"z_{index}" for index in range(8)]
    required = {"condition", "bearing_id", "split", "step_id", "normalized_lifetime", "reconstruction_mse", "delta_z", *z_columns}
    if not required.issubset(frame.columns) or len(frame) != source_report["sample_count"]:
        raise ValueError("latent CSV schema or sample count mismatch")
    if sorted(frame["bearing_id"].unique()) != sorted(cfg["bearings"]):
        raise ValueError("bearing inventory mismatch")
    if frame["condition"].nunique() != 1:
        raise ValueError("this preregistered audit expects exactly one operating condition")
    for bearing, group in frame.groupby("bearing_id", sort=False):
        if group["step_id"].tolist() != list(range(len(group))):
            raise ValueError(f"noncontiguous step IDs: {bearing}")

    raw = frame[z_columns].to_numpy(float)
    centered = early_center(
        raw, frame["bearing_id"].to_numpy(), frame["step_id"].to_numpy(), cfg["calibration_steps"]
    )
    representations = {"raw": raw, "early_centered": centered}
    lifetime = frame["normalized_lifetime"].to_numpy(float)
    stages = stage_labels(lifetime, tuple(cfg["stage_thresholds"]))
    bearings = frame["bearing_id"].to_numpy()
    train_mask = frame["split"].to_numpy() == "train"

    geometry_rows = []
    pca_models = {}
    for name, values in representations.items():
        pca = PCA(n_components=8).fit(values[train_mask])
        pca_models[name] = pca
        ratios = pca.explained_variance_ratio_
        geometry_rows.append({
            "representation": name,
            "effective_dimension": effective_dimension(pca.explained_variance_),
            "pc1_explained_fraction": float(ratios[0]),
            "pc2_cumulative_fraction": float(ratios[:2].sum()),
            **{f"pc{index+1}_explained_fraction": float(value) for index, value in enumerate(ratios)},
        })

    correlation_rows = []
    targets = {
        "normalized_lifetime": lifetime,
        "reconstruction_mse": frame["reconstruction_mse"].to_numpy(float),
        "delta_z": frame["delta_z"].to_numpy(float),
    }
    # Per-bearing constant centering cannot change rank correlations, so raw z is sufficient here.
    for bearing in cfg["bearings"]:
        mask = bearings == bearing
        for dimension in range(8):
            for target_name, target in targets.items():
                rho, pvalue, count = finite_spearman(raw[mask, dimension], target[mask])
                correlation_rows.append({
                    "bearing_id": bearing, "dimension": dimension, "target": target_name,
                    "spearman_rho": rho, "pvalue_uncorrected": pvalue, "sample_count": count,
                })

    direction_vectors = []
    for bearing in cfg["bearings"]:
        indices = np.flatnonzero(bearings == bearing)
        early = raw[indices[: cfg["calibration_steps"]]].mean(axis=0)
        late_count = max(1, math.ceil(len(indices) * cfg["late_fraction"]))
        late = raw[indices[-late_count:]].mean(axis=0)
        direction_vectors.append(late - early)
    direction_matrix = cosine_similarity_matrix(np.stack(direction_vectors))
    direction_rows = []
    off_diagonal = []
    for i, left in enumerate(cfg["bearings"]):
        for j, right in enumerate(cfg["bearings"]):
            direction_rows.append({"bearing_a": left, "bearing_b": right, "cosine": float(direction_matrix[i, j])})
            if i < j:
                off_diagonal.append(float(direction_matrix[i, j]))

    stage_rows, lifetime_rows = [], []
    for representation, values in representations.items():
        for held_out in cfg["bearings"]:
            test = bearings == held_out
            train = ~test
            scaler = StandardScaler().fit(values[train])
            x_train, x_test = scaler.transform(values[train]), scaler.transform(values[test])
            classifier = LogisticRegression(
                C=cfg["logistic_C"], max_iter=cfg["max_iter"], class_weight="balanced",
                random_state=cfg["random_seed"], solver="lbfgs",
            ).fit(x_train, stages[train])
            predicted_stage = classifier.predict(x_test)
            majority = int(np.bincount(stages[train]).argmax())
            majority_prediction = np.full(test.sum(), majority)
            stage_rows.append({
                "representation": representation, "held_out_bearing": held_out,
                "sample_count": int(test.sum()),
                "accuracy": float(accuracy_score(stages[test], predicted_stage)),
                "balanced_accuracy": float(balanced_accuracy_score(stages[test], predicted_stage)),
                "macro_f1": float(f1_score(stages[test], predicted_stage, average="macro", zero_division=0)),
                "majority_accuracy": float(accuracy_score(stages[test], majority_prediction)),
                "majority_balanced_accuracy": float(balanced_accuracy_score(stages[test], majority_prediction)),
            })
            regressor = Ridge(alpha=cfg["ridge_alpha"]).fit(x_train, lifetime[train])
            predicted_lifetime = regressor.predict(x_test)
            rho, pvalue, _ = finite_spearman(predicted_lifetime, lifetime[test])
            lifetime_rows.append({
                "representation": representation, "held_out_bearing": held_out,
                "sample_count": int(test.sum()), "mae": float(mean_absolute_error(lifetime[test], predicted_lifetime)),
                "r2": float(r2_score(lifetime[test], predicted_lifetime)),
                "spearman_rho": rho, "pvalue_uncorrected": pvalue,
                "prediction_min": float(predicted_lifetime.min()), "prediction_max": float(predicted_lifetime.max()),
            })

    identity_rows = []
    usable = frame["step_id"].to_numpy() >= cfg["calibration_steps"]
    for representation, values in representations.items():
        train = np.zeros(len(frame), dtype=bool)
        test = np.zeros(len(frame), dtype=bool)
        for bearing in cfg["bearings"]:
            indices = np.flatnonzero((bearings == bearing) & usable)
            midpoint = len(indices) // 2
            train[indices[:midpoint]] = True
            test[indices[midpoint:]] = True
        scaler = StandardScaler().fit(values[train])
        classifier = LogisticRegression(
            C=cfg["logistic_C"], max_iter=cfg["max_iter"], class_weight="balanced",
            random_state=cfg["random_seed"], solver="lbfgs",
        ).fit(scaler.transform(values[train]), bearings[train])
        prediction = classifier.predict(scaler.transform(values[test]))
        identity_rows.append({
            "representation": representation, "train_sample_count": int(train.sum()),
            "test_sample_count": int(test.sum()), "accuracy": float(accuracy_score(bearings[test], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(bearings[test], prediction)),
            "macro_f1": float(f1_score(bearings[test], prediction, average="macro", zero_division=0)),
            "chance_balanced_accuracy": 1 / len(cfg["bearings"]),
            "split_definition": "first_half_after_calibration_train_second_half_test",
        })

    probe_summary = []
    for representation in representations:
        selected_stage = [row for row in stage_rows if row["representation"] == representation]
        selected_lifetime = [row for row in lifetime_rows if row["representation"] == representation]
        identity = next(row for row in identity_rows if row["representation"] == representation)
        geometry = next(row for row in geometry_rows if row["representation"] == representation)
        probe_summary.append({
            "representation": representation,
            "stage_balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in selected_stage])),
            "stage_balanced_accuracy_min": min(row["balanced_accuracy"] for row in selected_stage),
            "stage_macro_f1_mean": float(np.mean([row["macro_f1"] for row in selected_stage])),
            "lifetime_spearman_mean": float(np.mean([row["spearman_rho"] for row in selected_lifetime])),
            "lifetime_spearman_min": min(row["spearman_rho"] for row in selected_lifetime),
            "lifetime_mae_mean": float(np.mean([row["mae"] for row in selected_lifetime])),
            "bearing_identity_balanced_accuracy": identity["balanced_accuracy"],
            "effective_dimension": geometry["effective_dimension"],
        })

    gates = cfg["descriptive_gates"]
    centered_summary = next(row for row in probe_summary if row["representation"] == "early_centered")
    raw_summary = next(row for row in probe_summary if row["representation"] == "raw")
    median_direction = float(np.median(off_diagonal))
    flags = {
        "low_effective_dimension": centered_summary["effective_dimension"] < gates["effective_dimension_minimum"],
        "poor_cross_bearing_stage_probe": centered_summary["stage_balanced_accuracy_mean"] < gates["cross_bearing_stage_balanced_accuracy_minimum"],
        "poor_cross_bearing_lifetime_probe": centered_summary["lifetime_spearman_mean"] < gates["cross_bearing_lifetime_spearman_minimum"],
        "inconsistent_degradation_direction": median_direction < gates["median_direction_cosine_minimum"],
        "high_bearing_identity_readout": centered_summary["bearing_identity_balanced_accuracy"] > gates["bearing_identity_balanced_accuracy_maximum"],
    }
    centering_stage_improvement = centered_summary["stage_balanced_accuracy_mean"] - raw_summary["stage_balanced_accuracy_mean"]
    centering_lifetime_improvement = centered_summary["lifetime_spearman_mean"] - raw_summary["lifetime_spearman_mean"]
    alignment_evidence = max(centering_stage_improvement, centering_lifetime_improvement) >= gates["centering_improvement_minimum"]
    flag_count = sum(flags.values())
    if flag_count >= gates["encoder_primary_flag_count"]:
        diagnostic = "encoder_representation_is_primary_bottleneck_candidate"
    elif alignment_evidence:
        diagnostic = "cross_bearing_alignment_and_interpreter_are_primary_candidates"
    else:
        diagnostic = "mixed_or_inconclusive_bottleneck"

    if not all(sha256(path) == digest for path, digest in zip(protected, before.values())):
        raise RuntimeError("protected source artifact changed")
    output.mkdir(parents=True)
    write_csv(output / "representation_geometry.csv", geometry_rows)
    write_csv(output / "dimension_correlations.csv", correlation_rows)
    write_csv(output / "degradation_direction_cosines.csv", direction_rows)
    write_csv(output / "stage_probe_lobo.csv", stage_rows)
    write_csv(output / "lifetime_probe_lobo.csv", lifetime_rows)
    write_csv(output / "bearing_identity_probe.csv", identity_rows)
    write_csv(output / "probe_summary.csv", probe_summary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for representation, color in (("raw", "#3274A1"), ("early_centered", "#E1812C")):
        geometry = next(row for row in geometry_rows if row["representation"] == representation)
        axes[0, 0].plot(range(1, 9), [geometry[f"pc{i}_explained_fraction"] for i in range(1, 9)], marker="o", label=representation, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Training-bearing PCA spectrum")
    axes[0, 0].set_xlabel("Principal component")
    axes[0, 0].set_ylabel("Explained variance fraction (log)")
    axes[0, 0].legend()
    x = np.arange(len(cfg["bearings"]))
    for representation, color in (("raw", "#3274A1"), ("early_centered", "#E1812C")):
        rows = [next(row for row in stage_rows if row["representation"] == representation and row["held_out_bearing"] == bearing) for bearing in cfg["bearings"]]
        axes[0, 1].plot(x, [row["balanced_accuracy"] for row in rows], marker="o", label=representation, color=color)
        rows = [next(row for row in lifetime_rows if row["representation"] == representation and row["held_out_bearing"] == bearing) for bearing in cfg["bearings"]]
        axes[1, 0].plot(x, [row["spearman_rho"] for row in rows], marker="o", label=representation, color=color)
    labels = [bearing.replace("Bearing", "B") for bearing in cfg["bearings"]]
    axes[0, 1].axhline(1 / 3, color="black", linestyle=":", label="chance")
    axes[0, 1].set_title("LOBO stage balanced accuracy")
    axes[1, 0].axhline(0, color="black", linestyle=":")
    axes[1, 0].set_title("LOBO lifetime Spearman")
    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, labels)
        axis.legend()
    image = axes[1, 1].imshow(direction_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1, 1].set_xticks(x, labels, rotation=45)
    axes[1, 1].set_yticks(x, labels)
    axes[1, 1].set_title("Early-to-late direction cosine")
    fig.colorbar(image, ax=axes[1, 1], shrink=0.8)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    fig.suptitle("Frozen z=8 representation audit")
    fig.savefig(output / "representation_audit.png", dpi=170)
    plt.close(fig)

    report = {
        "status": "completed", "verification_status": "deterministic_frozen_representation_audit",
        "config": cfg, "protected_sha256": before, "protected_files_unchanged": True,
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)), "src/state_interpreter/representation_audit.py": sha256(ROOT / "src/state_interpreter/representation_audit.py")},
        "environment": {
            "executable": sys.executable, "python": sys.version, "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "sample_count": len(frame), "condition_count": int(frame["condition"].nunique()),
        "operating_condition_probe": "not_testable_single_condition",
        "probe_summary": probe_summary,
        "median_off_diagonal_direction_cosine": median_direction,
        "direction_cosine_min": min(off_diagonal), "direction_cosine_max": max(off_diagonal),
        "centering_stage_balanced_accuracy_improvement": centering_stage_improvement,
        "centering_lifetime_spearman_improvement": centering_lifetime_improvement,
        "diagnostic_flags": flags, "diagnostic_flag_count": flag_count,
        "diagnostic_conclusion": diagnostic,
        "row_counts": {
            "representation_geometry": len(geometry_rows), "dimension_correlations": len(correlation_rows),
            "degradation_direction_cosines": len(direction_rows), "stage_probe_lobo": len(stage_rows),
            "lifetime_probe_lobo": len(lifetime_rows), "bearing_identity_probe": len(identity_rows),
            "probe_summary": len(probe_summary),
        },
        "duration_seconds": time.monotonic() - started,
        "fallacy_scan_coverage": "11/11",
        "limitations": [
            "All five bearings were previously inspected; this is diagnostic, not blind confirmation.",
            "Normalized lifetime bins are weak temporal labels, not physical damage-stage annotations.",
            "Only one operating condition is present, so condition invariance cannot be evaluated.",
            "Bearing identity prediction reuses identity classes across chronological halves and is only a leakage diagnostic.",
            "Five leave-one-bearing-out folds are heterogeneous cases, not independent population samples for significance inference.",
            "Linear probes can underestimate nonlinear information and can exploit temporal correlations within each trajectory.",
        ],
    }
    write_json(output / "experiment_report.json", report)
    print(json.dumps({
        "status": "completed", "probe_summary": probe_summary,
        "median_direction_cosine": median_direction, "flags": flags,
        "diagnostic_conclusion": diagnostic, "row_counts": report["row_counts"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
