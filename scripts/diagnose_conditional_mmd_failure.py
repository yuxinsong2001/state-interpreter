"""Read-only diagnosis of the frozen conditional-MMD development CSV outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/2026-09-19_source_conditional_mmd_development_v1"
OUTPUT = ROOT / "results/2026-09-19_conditional_mmd_failure_diagnosis_v1"
ARMS = ("source_only", "global_mmd", "conditional_mmd")
BEARINGS = ("Bearing3_1", "Bearing3_2", "Bearing3_3")
SEEDS = (20260918, 20260921, 20260924)
INPUTS = ("experiment_report.json", "health_trajectories.csv", "per_seed_health_metrics.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def block_of(lifetime: float) -> int:
    if not 0 <= lifetime <= 1:
        raise ValueError("lifetime outside [0,1]")
    return min(int(lifetime * 5), 4)


def main() -> None:
    if OUTPUT.exists() or OUTPUT.with_name(OUTPUT.name + ".staging").exists():
        raise FileExistsError("diagnosis output already exists")
    report = json.loads((SOURCE / INPUTS[0]).read_text(encoding="utf-8"))
    if report["combined_gate_passed"] or report["protected_bearings_not_read"] != ["Bearing3_4", "Bearing3_5"]:
        raise ValueError("unexpected development gate/protection status")
    if report["config_sha256"] != "1d9e9cfbf8b5f04bc2beaccdd5fd18fa8abf1a57b1efbef1c17cc921b826114b":
        raise ValueError("unexpected source config")

    grouped: dict[tuple[str, int, str], list[tuple[int, float, float]]] = defaultdict(list)
    for row in read_csv(SOURCE / INPUTS[1]):
        key = (row["arm"], int(row["seed"]), row["bearing_id"])
        if key[0] not in ARMS or key[1] not in SEEDS or key[2] not in BEARINGS:
            raise ValueError("trajectory outside development scope")
        item = (int(row["step_id"]), float(row["normalized_lifetime"]), float(row["predicted_health"]))
        if not np.isfinite(item[1]) or not np.isfinite(item[2]):
            raise ValueError("nonfinite trajectory")
        grouped[key].append(item)
    if len(grouped) != 27:
        raise ValueError("missing arm/seed/bearing combination")

    metrics = {(r["arm"], int(r["seed"]), r["outer_test_bearing"]): float(r["spearman_health"])
               for r in read_csv(SOURCE / INPUTS[2])}
    stage_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    colors = {20260918: "#1f77b4", 20260921: "#ff7f0e", 20260924: "#2ca02c"}

    for bi, bearing in enumerate(BEARINGS):
        for ai, arm in enumerate(ARMS):
            ax = axes[bi, ai]
            for seed in SEEDS:
                key = (arm, seed, bearing)
                rows = sorted(grouped[key])
                steps = [r[0] for r in rows]
                life = np.asarray([r[1] for r in rows])
                pred = np.asarray([r[2] for r in rows])
                if steps != list(range(len(rows))) or len(rows) < 20:
                    raise ValueError("incomplete/noncontiguous trajectory")
                rho = float(spearmanr(life, pred).statistic)
                if not np.isclose(rho, metrics[key], atol=1e-10):
                    raise ValueError("saved Spearman mismatch")
                for block in range(5):
                    indices = np.asarray([block_of(float(x)) == block for x in life])
                    stage_rows.append({
                        "arm": arm, "seed": seed, "bearing_id": bearing, "time_block": block,
                        "n": int(indices.sum()),
                        "within_block_spearman": float(spearmanr(life[indices], pred[indices]).statistic),
                        "mean_prediction": float(pred[indices].mean()),
                        "median_prediction": float(np.median(pred[indices])),
                    })
                means = [r["mean_prediction"] for r in stage_rows[-5:]]
                trajectory_rows.append({
                    "arm": arm, "seed": seed, "bearing_id": bearing, "full_spearman": rho,
                    "first_to_last_block_mean_change": means[-1] - means[0],
                    "first_to_last_block_direction": "up" if means[-1] > means[0] else "down",
                    "adjacent_block_mean_increases": sum(b > a for a, b in zip(means, means[1:])),
                })
                every = max(1, len(rows) // 300)
                ax.plot(life[::every], pred[::every], color=colors[seed], alpha=0.7,
                        linewidth=0.8, label=str(seed))
            ax.set_title(f"{bearing} / {arm}")
            ax.set_xlabel("normalized lifetime")
            ax.set_ylabel("predicted health")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
    fig.suptitle("Condition 3 development trajectories: raw predictions, no smoothing")
    fig.tight_layout()

    paired_rows = []
    by_key = {(r["arm"], r["seed"], r["bearing_id"]): r for r in trajectory_rows}
    for bearing in BEARINGS:
        for seed in SEEDS:
            base = by_key[("source_only", seed, bearing)]
            for arm in ARMS[1:]:
                current = by_key[(arm, seed, bearing)]
                paired_rows.append({
                    "arm": arm, "seed": seed, "bearing_id": bearing,
                    "full_spearman_minus_source_only": current["full_spearman"] - base["full_spearman"],
                    "block_mean_change_minus_source_only": current["first_to_last_block_mean_change"] - base["first_to_last_block_mean_change"],
                })

    summary = []
    for bearing in BEARINGS:
        for arm in ARMS:
            subset = [r for r in trajectory_rows if r["bearing_id"] == bearing and r["arm"] == arm]
            summary.append({
                "bearing_id": bearing, "arm": arm,
                "mean_full_spearman": mean(r["full_spearman"] for r in subset),
                "positive_full_spearman_seeds": sum(r["full_spearman"] > 0 for r in subset),
                "mean_first_to_last_block_change": mean(r["first_to_last_block_mean_change"] for r in subset),
                "upward_first_to_last_block_seeds": sum(r["first_to_last_block_mean_change"] > 0 for r in subset),
            })

    staging = OUTPUT.with_name(OUTPUT.name + ".staging")
    staging.mkdir(parents=True)
    write_csv(staging / "within_block_metrics.csv", stage_rows)
    write_csv(staging / "trajectory_summary_by_seed.csv", trajectory_rows)
    write_csv(staging / "paired_delta_by_seed.csv", paired_rows)
    write_csv(staging / "summary_by_bearing_arm.csv", summary)
    fig.savefig(staging / "development_trajectories.png", dpi=160)
    plt.close(fig)
    audit = {
        "status": "exploratory_read_only_diagnosis_completed",
        "source_experiment": report["experiment_id"],
        "source_sha256": {name: sha256(SOURCE / name) for name in INPUTS},
        "source_bearings_only": list(BEARINGS),
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "trajectories": len(trajectory_rows), "within_block_rows": len(stage_rows),
        "paired_rows": len(paired_rows),
        "warnings": ["time blocks are measurement-progress bins, not verified damage stages",
                     "diagnosis is exploratory and cannot re-open preregistered gate"],
    }
    (staging / "diagnosis_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    staging.rename(OUTPUT)
    print(json.dumps({"audit": audit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
