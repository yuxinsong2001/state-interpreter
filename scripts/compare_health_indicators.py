"""Compare interpretable health-state candidates from ordered latent vectors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def early_count(length: int, fraction: float) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return max(1, math.ceil(length * fraction))


def self_relative_distance(z: np.ndarray, fraction: float) -> np.ndarray:
    """Distance from the mean of this episode's early calibration window."""

    if z.ndim != 2 or len(z) == 0:
        raise ValueError("z must have shape [samples, latent_dim]")
    baseline = z[: early_count(len(z), fraction)].mean(axis=0)
    return np.linalg.norm(z - baseline, axis=1)


def causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Average only the current and previous values."""

    if window <= 0:
        raise ValueError("window must be positive")
    result = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def causal_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope over the current causal window; zero until two points exist."""

    if window <= 1:
        raise ValueError("window must be greater than one")
    result = np.zeros(len(values), dtype=np.float64)
    for index in range(1, len(values)):
        start = max(0, index - window + 1)
        y = values[start : index + 1]
        x = np.arange(len(y), dtype=np.float64)
        result[index] = float(np.polyfit(x, y, deg=1)[0])
    return result


def monotonicity(values: np.ndarray) -> float:
    """Absolute balance of positive and negative first differences."""

    differences = np.diff(values)
    nonzero = np.sign(differences[np.abs(differences) > 1e-12])
    return 0.0 if len(nonzero) == 0 else float(abs(nonzero.mean()))


def roughness_smoothness(values: np.ndarray) -> float:
    """Custom [0,1] smoothness based on second/first-difference roughness."""

    first = np.diff(values)
    if len(first) < 2:
        return 1.0
    ratio = np.sum(np.abs(np.diff(first))) / (np.sum(np.abs(first)) + 1e-12)
    return float(1.0 / (1.0 + ratio))


def resample(values: np.ndarray, points: int = 100) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, values)


def cross_bearing_metrics(curves: list[np.ndarray]) -> dict[str, float]:
    correlations: list[float] = []
    resampled = [resample(curve) for curve in curves]
    for left in range(len(resampled)):
        for right in range(left + 1, len(resampled)):
            correlation = float(pearsonr(resampled[left], resampled[right]).statistic)
            correlations.append(correlation)
    starts = np.asarray([curve[0] for curve in curves])
    ends = np.asarray([curve[-1] for curve in curves])
    change_scale = float(np.mean(np.abs(ends - starts)))
    prognosability = float(np.exp(-np.std(ends) / (change_scale + 1e-12)))
    return {
        "trendability_min_absolute_pairwise_correlation": float(
            min(abs(value) for value in correlations)
        ),
        "trend_consistency_min_signed_pairwise_correlation": float(min(correlations)),
        "prognosability": prognosability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--healthy-fraction", type=float, default=0.10)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    if args.window <= 1:
        raise ValueError("window must be greater than one")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    with Path(args.input).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV contains no rows")
    z_fields = sorted(
        (name for name in rows[0] if name.startswith("z_")),
        key=lambda name: int(name.split("_")[1]),
    )
    if not z_fields:
        raise ValueError("input CSV contains no latent columns")

    bearing_order = list(dict.fromkeys(row["bearing_id"] for row in rows))
    by_bearing: dict[str, list[dict[str, str]]] = {
        bearing: [row for row in rows if row["bearing_id"] == bearing]
        for bearing in bearing_order
    }
    features: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    curves_by_indicator: dict[str, list[np.ndarray]] = {
        "global_distance": [],
        "self_relative_distance": [],
        "causal_smoothed_relative_distance": [],
    }

    for bearing, bearing_rows in by_bearing.items():
        steps = np.asarray([int(row["step_id"]) for row in bearing_rows])
        if not np.all(np.diff(steps) > 0):
            raise ValueError(f"steps are not strictly increasing for {bearing}")
        z = np.asarray(
            [[float(row[field]) for field in z_fields] for row in bearing_rows]
        )
        global_distance = np.asarray(
            [float(row["health_center_distance"]) for row in bearing_rows]
        )
        delta_z = np.asarray(
            [0.0 if row["delta_z"] == "" else float(row["delta_z"]) for row in bearing_rows]
        )
        relative = self_relative_distance(z, args.healthy_fraction)
        smoothed = causal_moving_average(relative, args.window)
        slope = causal_slope(smoothed, args.window)
        normalized_lifetime = np.asarray(
            [float(row["normalized_lifetime"]) for row in bearing_rows]
        )

        indicators = {
            "global_distance": global_distance,
            "self_relative_distance": relative,
            "causal_smoothed_relative_distance": smoothed,
        }
        for indicator_name, values in indicators.items():
            rho = float(spearmanr(normalized_lifetime, values).statistic)
            metric_rows.append(
                {
                    "bearing_id": bearing,
                    "split": bearing_rows[0]["split"],
                    "indicator": indicator_name,
                    "spearman_rho": rho,
                    "monotonicity": monotonicity(values),
                    "smoothness": roughness_smoothness(values),
                    "first_value": float(values[0]),
                    "last_value": float(values[-1]),
                }
            )
            curves_by_indicator[indicator_name].append(values)

        for index, row in enumerate(bearing_rows):
            features.append(
                {
                    "bearing_id": bearing,
                    "split": row["split"],
                    "step_id": int(row["step_id"]),
                    "normalized_lifetime": float(row["normalized_lifetime"]),
                    "global_distance": float(global_distance[index]),
                    "self_relative_distance": float(relative[index]),
                    "state_level": float(smoothed[index]),
                    "state_trend": float(slope[index]),
                    "state_movement": float(delta_z[index]),
                }
            )

    with (output_dir / "state_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    with (output_dir / "indicator_bearing_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary: dict[str, object] = {}
    for indicator_name, curves in curves_by_indicator.items():
        selected = [row for row in metric_rows if row["indicator"] == indicator_name]
        rhos = np.asarray([float(row["spearman_rho"]) for row in selected])
        summary[indicator_name] = {
            "mean_spearman_rho": float(rhos.mean()),
            "minimum_spearman_rho": float(rhos.min()),
            "positive_bearings": int((rhos > 0).sum()),
            "mean_monotonicity": float(
                np.mean([float(row["monotonicity"]) for row in selected])
            ),
            "mean_smoothness": float(
                np.mean([float(row["smoothness"]) for row in selected])
            ),
            **cross_bearing_metrics(curves),
        }

    report = {
        "status": "completed",
        "input": str(Path(args.input).resolve()),
        "sample_count": len(rows),
        "bearing_count": len(bearing_order),
        "healthy_fraction": args.healthy_fraction,
        "causal_window": args.window,
        "candidate_states": {
            "A": ["global_distance"],
            "B": ["self_relative_distance"],
            "C": ["state_level", "state_trend", "state_movement"],
        },
        "indicator_summary": summary,
        "metric_notes": {
            "trendability": "minimum absolute pairwise Pearson correlation after lifetime resampling",
            "trend_consistency": "minimum signed pairwise Pearson correlation; negative values expose reversed trajectories",
            "prognosability": "exp(-std(end values)/mean absolute end-start change)",
            "smoothness": "custom roughness score 1/(1 + second-difference/first-difference ratio)",
            "warning": "normalized lifetime is a time proxy, not a measured damage label",
        },
        "interpretation_gate": {
            "passed": False,
            "reason": "Metrics generated; selection requires scientific interpretation.",
        },
    }
    with (output_dir / "comparison_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=False)
    titles = {
        "global_distance": "A: Global healthy-center distance",
        "self_relative_distance": "B: Per-bearing early-baseline distance",
        "causal_smoothed_relative_distance": "C level: Causal-smoothed relative distance",
    }
    for axis, (indicator_name, curves) in zip(axes, curves_by_indicator.items()):
        for bearing, values in zip(bearing_order, curves):
            axis.plot(np.arange(len(values)), values, label=bearing, linewidth=1.5)
        axis.set_title(titles[indicator_name])
        axis.set_ylabel("Indicator value")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Measurement step")
    fig.tight_layout()
    fig.savefig(output_dir / "indicator_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    state_fields = ["state_level", "state_trend", "state_movement"]
    labels = ["Causal level", "Recent slope", "Adjacent Δz"]
    for axis, field, label in zip(axes, state_fields, labels):
        for bearing in bearing_order:
            selected = [row for row in features if row["bearing_id"] == bearing]
            axis.plot(
                [int(row["step_id"]) for row in selected],
                [float(row[field]) for row in selected],
                label=bearing,
                linewidth=1.5,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Measurement step")
    fig.suptitle("Candidate C: interpretable temporal state")
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_state_features.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
