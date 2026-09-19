"""Exploratory stage diagnosis of immutable Source-DANN development outputs.

Reads only the prior experiment's CSV/JSON files; it does not load datasets or
checkpoints, train models, or change the locked experiment configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results" / "2026-09-18_source_dann_development_v1"
OUTPUT = ROOT / "results" / "2026-09-19_source_dann_lifecycle_diagnosis_v1"
INPUT_FILES = (
    "experiment_report.json",
    "health_trajectories.csv",
    "identity_probe.csv",
    "per_seed_health_metrics.csv",
)
ARMS = ("source_only", "source_dann")
BEARINGS = ("Bearing3_1", "Bearing3_2", "Bearing3_3")
SEEDS = (20260918, 20260921, 20260924)
BLOCKS = 5


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lifecycle_block(lifetime: float) -> int:
    if not 0.0 <= lifetime <= 1.0:
        raise ValueError("normalized lifetime outside [0, 1]")
    return min(int(lifetime * BLOCKS), BLOCKS - 1)


def paired_stage_health(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[int, str, int, str], list[tuple[int, float, float]]] = {}
    for row in rows:
        arm = row["arm"]
        seed = int(row["seed"])
        bearing = row["bearing_id"]
        step = int(row["step_id"])
        life = float(row["normalized_lifetime"])
        predicted = float(row["predicted_health"])
        if arm not in ARMS or seed not in SEEDS or bearing not in BEARINGS:
            raise ValueError("trajectory outside fixed development scope")
        if not np.isfinite(predicted):
            raise ValueError("non-finite health prediction")
        grouped.setdefault((seed, bearing, lifecycle_block(life), arm), []).append(
            (step, life, predicted)
        )

    output = []
    for seed in SEEDS:
        for bearing in BEARINGS:
            for block in range(BLOCKS):
                base = sorted(grouped[(seed, bearing, block, "source_only")])
                dann = sorted(grouped[(seed, bearing, block, "source_dann")])
                if len(base) < 3 or len(base) != len(dann):
                    raise ValueError("incomplete paired stage")
                if [item[:2] for item in base] != [item[:2] for item in dann]:
                    raise ValueError("paired stage timestamps differ")
                lifetime = np.asarray([item[1] for item in base])
                base_prediction = np.asarray([item[2] for item in base])
                dann_prediction = np.asarray([item[2] for item in dann])
                rho_base = float(spearmanr(lifetime, base_prediction).statistic)
                rho_dann = float(spearmanr(lifetime, dann_prediction).statistic)
                output.append(
                    {
                        "seed": seed,
                        "bearing_id": bearing,
                        "lifetime_block": block,
                        "lifetime_start": block / BLOCKS,
                        "lifetime_end": (block + 1) / BLOCKS,
                        "measurement_count": len(base),
                        "source_only_within_block_spearman": rho_base,
                        "source_dann_within_block_spearman": rho_dann,
                        "dann_minus_source_only": rho_dann - rho_base,
                        "source_only_mean_health": float(base_prediction.mean()),
                        "source_dann_mean_health": float(dann_prediction.mean()),
                    }
                )
    return output


def paired_stage_identity(rows: list[dict[str, str]]) -> list[dict]:
    indexed: dict[tuple[int, str, int, str], dict[str, str]] = {}
    for row in rows:
        if row["held_lifetime_block"] == "all":
            continue
        key = (
            int(row["seed"]),
            row["outer_test_bearing"],
            int(row["held_lifetime_block"]),
            row["arm"],
        )
        if key in indexed:
            raise ValueError("duplicate identity block")
        indexed[key] = row
    output = []
    for seed in SEEDS:
        for bearing in BEARINGS:
            for block in range(BLOCKS):
                base = indexed[(seed, bearing, block, "source_only")]
                dann = indexed[(seed, bearing, block, "source_dann")]
                if base["test_samples"] != dann["test_samples"]:
                    raise ValueError("identity block sample counts differ")
                accuracy_base = float(base["accuracy"])
                accuracy_dann = float(dann["accuracy"])
                output.append(
                    {
                        "seed": seed,
                        "outer_test_bearing": bearing,
                        "lifetime_block": block,
                        "probe_sample_count": int(base["test_samples"]),
                        "source_only_identity_accuracy": accuracy_base,
                        "source_dann_identity_accuracy": accuracy_dann,
                        "identity_reduction": accuracy_base - accuracy_dann,
                    }
                )
    return output


def summarize(health: list[dict], identity: list[dict]) -> list[dict]:
    summary = []
    for bearing in BEARINGS:
        for block in range(BLOCKS):
            hs = [
                item for item in health
                if item["bearing_id"] == bearing and item["lifetime_block"] == block
            ]
            ids = [
                item for item in identity
                if item["outer_test_bearing"] == bearing
                and item["lifetime_block"] == block
            ]
            summary.append(
                {
                    "bearing_id": bearing,
                    "lifetime_block": block,
                    "source_only_mean_within_block_spearman": mean(
                        item["source_only_within_block_spearman"] for item in hs
                    ),
                    "source_dann_mean_within_block_spearman": mean(
                        item["source_dann_within_block_spearman"] for item in hs
                    ),
                    "mean_health_delta": mean(item["dann_minus_source_only"] for item in hs),
                    "positive_health_delta_seeds": sum(
                        item["dann_minus_source_only"] > 0 for item in hs
                    ),
                    "mean_identity_reduction": mean(
                        item["identity_reduction"] for item in ids
                    ),
                    "positive_identity_reduction_seeds": sum(
                        item["identity_reduction"] > 0 for item in ids
                    ),
                    "paired_seed_count": len(hs),
                }
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    if args.confirm != "DIAGNOSE_SOURCE_DANN_DEVELOPMENT_OUTPUTS_ONLY":
        raise PermissionError("exact diagnostic token required")
    if OUTPUT.exists() or OUTPUT.with_name(OUTPUT.name + ".staging").exists():
        raise FileExistsError("refusing to overwrite diagnosis output")
    report = json.loads((EXPERIMENT / "experiment_report.json").read_text(encoding="utf-8"))
    if report["combined_gate_passed"] or report["validation_authorized"]:
        raise ValueError("unexpected experiment gate status")
    if report["protected_bearings_not_read"] != ["Bearing3_4", "Bearing3_5"]:
        raise ValueError("protected bearing status changed")
    source_hashes = {name: digest(EXPERIMENT / name) for name in INPUT_FILES}
    health = paired_stage_health(read_csv(EXPERIMENT / "health_trajectories.csv"))
    identity = paired_stage_identity(read_csv(EXPERIMENT / "identity_probe.csv"))
    summary = summarize(health, identity)
    staging = OUTPUT.with_name(OUTPUT.name + ".staging")
    staging.mkdir(parents=True)
    write_csv(staging / "paired_stage_health.csv", health)
    write_csv(staging / "paired_stage_identity.csv", identity)
    write_csv(staging / "stage_summary.csv", summary)
    result = {
        "status": "exploratory_development_diagnosis_completed",
        "source_experiment": report["experiment_id"],
        "source_config_sha256": report["config_sha256"],
        "source_file_sha256": source_hashes,
        "five_equal_normalized_lifetime_blocks": True,
        "three_seeds": list(SEEDS),
        "three_development_bearings": list(BEARINGS),
        "paired_stage_health_rows": len(health),
        "paired_stage_identity_rows": len(identity),
        "stage_summary_rows": len(summary),
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "no_new_training": True,
        "verification_status": "ANALYZED",
    }
    (staging / "diagnosis_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    staging.rename(OUTPUT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
