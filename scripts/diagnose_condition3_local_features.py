"""Exploratory, read-only local-feature audit for Condition 3 development bearings."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler
from state_interpreter.features.bearing_features import FEATURE_NAMES

CONFIG = ROOT / "configs/xjtu_condition3_source_conditional_mmd_development_v1.json"
OUTPUT = ROOT / "results/2026-09-19_condition3_local_feature_audit_v1"
BEARINGS = ("Bearing3_1", "Bearing3_2", "Bearing3_3")
SELECTED = (
    "h_rms", "v_rms", "h_kurtosis", "v_kurtosis",
    "h_bpfo_band_power", "v_bpfo_band_power",
    "h_bpfi_band_power", "v_bpfi_band_power",
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(y) <= 1e-12:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def main() -> None:
    staging = OUTPUT.with_name(OUTPUT.name + ".staging")
    if OUTPUT.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite local feature audit")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["development_bearings"] != list(BEARINGS):
        raise ValueError("unexpected development scope")
    if config["protected_validation"] != ["Bearing3_4"] or config["protected_blind"] != ["Bearing3_5"]:
        raise ValueError("unexpected protected scope")
    if config["input"]["feature_count"] != len(FEATURE_NAMES) or config["input"]["scaling_mode"] != "signed_log1p":
        raise ValueError("unexpected feature/scaling protocol")
    if [x["bearing_id"] for x in config["input"]["cache_entries"]] != list(BEARINGS):
        raise ValueError("unexpected cache entries")
    if not set(SELECTED).issubset(FEATURE_NAMES):
        raise ValueError("selected feature not in feature catalog")

    cache = ROOT / config["input"]["cache_directory"]
    raw: dict[str, np.ndarray] = {}
    scaled: dict[str, np.ndarray] = {}
    cache_hashes = {}
    for entry in config["input"]["cache_entries"]:
        bearing = entry["bearing_id"]
        path = cache / f"{bearing}.npz"
        cache_hashes[bearing] = digest(path)
        if cache_hashes[bearing] != entry["sha256"]:
            raise ValueError(f"cache hash mismatch: {bearing}")
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
            raise ValueError(f"invalid feature cache: {bearing}")
        raw[bearing] = features
        tensor = torch.from_numpy(features)
        scaler = StableCausalFeatureScaler.fit(
            tensor, calibration_steps=config["input"]["calibration_steps"],
            mode=config["input"]["scaling_mode"],
        )
        scaled[bearing] = scaler.transform(tensor).numpy()

    full_rows: list[dict] = []
    block_rows: list[dict] = []
    for bearing in BEARINGS:
        features = raw[bearing]
        normalized_time = np.arange(len(features)) / (len(features) - 1)
        for index, name in enumerate(FEATURE_NAMES):
            values = features[:, index]
            record = {"bearing_id": bearing, "feature": name, "steps": len(values),
                      "full_spearman": correlation(normalized_time, values)}
            full_rows.append(record)
            for block in range(5):
                mask = np.minimum((normalized_time * 5).astype(int), 4) == block
                block_rows.append({
                    "bearing_id": bearing, "feature": name, "time_block": block,
                    "steps": int(mask.sum()),
                    "within_block_spearman": correlation(normalized_time[mask], values[mask]),
                    "mean_early_scaled_value": float(scaled[bearing][mask, index].mean()),
                    "median_early_scaled_value": float(np.median(scaled[bearing][mask, index])),
                })

    selected_rows = [r for r in full_rows if r["feature"] in SELECTED]
    selected_blocks = [r for r in block_rows if r["feature"] in SELECTED]
    selected_names = list(SELECTED)
    fig, axes = plt.subplots(len(selected_names), len(BEARINGS), figsize=(14, 18), sharex=True)
    for fi, name in enumerate(selected_names):
        idx = FEATURE_NAMES.index(name)
        for bi, bearing in enumerate(BEARINGS):
            ax = axes[fi, bi]
            values = scaled[bearing][:, idx]
            time = np.arange(len(values)) / (len(values) - 1)
            stride = max(1, len(values) // 500)
            ax.plot(time[::stride], values[::stride], linewidth=0.7)
            ax.set_title(f"{bearing}: {name}", fontsize=9)
            ax.grid(alpha=0.2)
            if bi == 0:
                ax.set_ylabel("signed-log early z-score")
            if fi == len(selected_names) - 1:
                ax.set_xlabel("normalized lifetime")
    fig.suptitle("Selected Condition 3 features; visual downsampling only")
    fig.tight_layout()

    staging.mkdir(parents=True)
    write_csv(staging / "all_feature_full_spearman.csv", full_rows)
    write_csv(staging / "all_feature_five_block_metrics.csv", block_rows)
    write_csv(staging / "preselected_feature_full_spearman.csv", selected_rows)
    write_csv(staging / "preselected_feature_five_block_metrics.csv", selected_blocks)
    fig.savefig(staging / "preselected_feature_trajectories.png", dpi=140)
    plt.close(fig)
    audit = {
        "status": "exploratory_read_only_feature_audit_completed",
        "config_sha256": digest(CONFIG), "development_cache_sha256": cache_hashes,
        "source_bearings_only": list(BEARINGS),
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "feature_count": len(FEATURE_NAMES), "preselected_features": selected_names,
        "full_rows": len(full_rows), "block_rows": len(block_rows),
        "warnings": [
            "normalized lifetime is measurement progress, not physical damage",
            "feature associations are exploratory, not model attribution or causal effects",
            "do not select features or reopen development gate based on this audit",
        ],
    }
    (staging / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    staging.rename(OUTPUT)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
