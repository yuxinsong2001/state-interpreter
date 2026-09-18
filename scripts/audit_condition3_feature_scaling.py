"""Audit preregistered stable scaling candidates on development caches only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_interpreter.data.stable_feature_scaling import StableCausalFeatureScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_before_scaling_audit_and_before_training":
        raise ValueError("invalid scaling amendment status")
    if args.confirm != config["execution_token"]:
        raise PermissionError("exact scaling audit token required")
    expected_development = ["Bearing3_1", "Bearing3_2", "Bearing3_3"]
    if config["scope"]["development"] != expected_development:
        raise ValueError("development scope changed")
    scaler_path = ROOT / config["frozen_scaler"]["path"]
    if sha256(scaler_path) != config["frozen_scaler"]["sha256"]:
        raise ValueError("frozen stable scaler hash mismatch")

    output = Path(args.output).resolve()
    expected_output = (ROOT / "results/2026-09-18_feature_scaling_audit").resolve()
    if output != expected_output:
        raise ValueError(f"output must equal {expected_output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scaling audit: {output}")

    cache = (ROOT / config["input_cache"]["directory"]).resolve()
    arrays: dict[str, torch.Tensor] = {}
    for entry in config["input_cache"]["entries"]:
        bearing = entry["bearing_id"]
        if bearing not in expected_development:
            raise ValueError("cache entry outside development scope")
        path = cache / f"{bearing}.npz"
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"cache hash mismatch: {bearing}")
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["features"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 65:
            raise ValueError(f"invalid feature shape for {bearing}: {values.shape}")
        arrays[bearing] = torch.from_numpy(values)

    rows: list[dict[str, object]] = []
    gate = config["selection_gate"]
    for candidate in config["candidates"]:
        mode = candidate["mode"]
        for bearing in expected_development:
            scaler = StableCausalFeatureScaler.fit(
                arrays[bearing],
                calibration_steps=config["fixed_calibration_steps"],
                mode=mode,
            )
            transformed = scaler.transform(arrays[bearing]).numpy()
            absolute = np.abs(transformed)
            maximum = float(absolute.max())
            p99 = float(np.quantile(absolute, 0.99))
            rows.append(
                {
                    "mode": mode,
                    "bearing_id": bearing,
                    "finite": bool(np.isfinite(transformed).all()),
                    "max_abs": maximum,
                    "p99_abs": p99,
                    "strictly_monotonic_transform": bool(
                        candidate["strictly_monotonic"]
                    ),
                    "passes_gate": bool(
                        np.isfinite(transformed).all()
                        and maximum <= gate["maximum_absolute_value_at_most"]
                        and p99 <= gate["p99_absolute_value_at_most"]
                        and (
                            candidate["strictly_monotonic"]
                            or not gate["strictly_monotonic_transform_required"]
                        )
                    ),
                }
            )

    selected = None
    for candidate in config["candidates"]:
        candidate_rows = [row for row in rows if row["mode"] == candidate["mode"]]
        if all(bool(row["passes_gate"]) for row in candidate_rows):
            selected = candidate["mode"]
            break

    output.mkdir(parents=True)
    with (output / "scaling_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "amendment_id": config["amendment_id"],
        "status": "scaling_audit_completed",
        "selected_mode": selected,
        "selection_gate": gate,
        "metrics": rows,
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "training_performed": False,
        "input_cache_hashes_verified": True,
        "scaler_hash_verified": True,
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
