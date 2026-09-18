"""Protected, development-only feature materialization for XJTU-SY."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from state_interpreter.data.feature_sequences import CausalFeatureStandardizer
from state_interpreter.feature_lstm_protocol import (
    EXPECTED_DEVELOPMENT,
    FeatureLSTMDevelopmentPolicy,
    sha256,
)
from state_interpreter.features.bearing_features import (
    FEATURE_NAMES,
    NUM_BEARING_FEATURES,
    XJTUBearingFeatureExtractor,
)


EXPECTED_HEADER = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)


def _ordered_csv_files(bearing_dir: Path) -> list[Path]:
    """Enumerate CSV files only inside an already-authorized bearing directory."""

    try:
        files = sorted(bearing_dir.glob("*.csv"), key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"non-numeric measurement filename in {bearing_dir}") from error
    if not files:
        raise ValueError(f"no CSV files in authorized bearing: {bearing_dir}")
    indices = [int(path.stem) for path in files]
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError(f"measurement indices are not consecutive in {bearing_dir}")
    return files


def _read_vibration(path: Path, *, expected_samples: int = 32768) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig") as stream:
        header = tuple(stream.readline().strip().split(","))
    if header != EXPECTED_HEADER:
        raise ValueError(f"unexpected CSV header in {path}: {header}")
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.shape != (expected_samples, 2):
        raise ValueError(
            f"expected {(expected_samples, 2)} samples in {path}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite vibration values in {path}")
    return values


def _quantiles(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    p50, p95, p99 = np.quantile(absolute, [0.50, 0.95, 0.99])
    return {
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(absolute.max()),
    }


def materialize_authorized_bearing(
    bearing_dir: Path,
    *,
    condition: str,
    calibration_steps: int,
    output_path: Path,
    expected_samples: int = 32768,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract one authorized bearing and write its immutable NPZ cache."""

    files = _ordered_csv_files(bearing_dir)
    extractor = XJTUBearingFeatureExtractor(condition)
    features = np.empty((len(files), NUM_BEARING_FEATURES), dtype=np.float32)
    for index, source in enumerate(files):
        features[index] = extractor.extract(
            _read_vibration(source, expected_samples=expected_samples)
        )
        if progress is not None and ((index + 1) % 100 == 0 or index + 1 == len(files)):
            progress(f"{bearing_dir.name}: {index + 1}/{len(files)}")

    step_ids = np.arange(len(files), dtype=np.int64)
    source_paths = np.asarray(
        [f"{condition}/{bearing_dir.name}/{path.name}" for path in files],
        dtype=np.str_,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        step_ids=step_ids,
        source_paths=source_paths,
    )

    tensor = torch.from_numpy(features)
    standardizer = CausalFeatureStandardizer.fit(
        tensor, calibration_steps=calibration_steps
    )
    calibrated = standardizer.transform(tensor).numpy()
    early_std = features[:calibration_steps].std(axis=0)
    full_std = features.std(axis=0)
    near_constant_indices = np.flatnonzero(early_std < 1e-8).tolist()
    entry = {
        "bearing_id": bearing_dir.name,
        "sample_count": len(files),
        "fields": ["features", "step_ids", "source_paths"],
        "path": output_path.name,
        "sha256": sha256(output_path),
    }
    quality = {
        "bearing_id": bearing_dir.name,
        "sample_count": len(files),
        "feature_shape": list(features.shape),
        "all_finite": bool(np.isfinite(features).all()),
        "step_ids": {
            "first": int(step_ids[0]),
            "last": int(step_ids[-1]),
            "unique": int(np.unique(step_ids).size),
            "consecutive": bool(np.array_equal(step_ids, np.arange(len(files)))),
        },
        "near_constant_early_feature_count": len(near_constant_indices),
        "near_constant_early_features": [
            FEATURE_NAMES[index] for index in near_constant_indices
        ],
        "full_sequence_zero_variance_feature_count": int(
            np.count_nonzero(full_std < 1e-12)
        ),
        "raw_absolute_quantiles": _quantiles(features),
        "early_calibrated_absolute_quantiles": _quantiles(calibrated),
    }
    return entry, quality


def materialize_development_cache(
    *,
    policy: FeatureLSTMDevelopmentPolicy,
    dataset_root: Path,
    token: str,
    output_dir: Path,
    expected_samples: int = 32768,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    """Materialize exactly B3_1--B3_3 without enumerating their parent directory."""

    # Authorization and frozen-code verification intentionally precede any dataset access.
    bearings = policy.authorize_materialization(list(EXPECTED_DEVELOPMENT), token)
    verified = policy.verify_frozen_artifacts()
    expected_output = (
        policy.repository_root / policy.config["materialization"]["cache_directory"]
    ).resolve()
    if output_dir.resolve() != expected_output:
        raise ValueError(f"output_dir must equal preregistered cache path: {expected_output}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output_dir}")

    condition = policy.config["condition"]
    condition_root = dataset_root.resolve() / condition
    calibration_steps = policy.config["features"]["calibration_steps"]
    output_dir.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for bearing in bearings:
        bearing_dir = condition_root / bearing
        if not bearing_dir.is_dir():
            raise FileNotFoundError(f"authorized bearing directory missing: {bearing_dir}")
        entry, quality = materialize_authorized_bearing(
            bearing_dir,
            condition=condition,
            calibration_steps=calibration_steps,
            output_path=output_dir / f"{bearing}.npz",
            expected_samples=expected_samples,
            progress=progress,
        )
        entries.append(entry)
        quality_rows.append(quality)

    manifest = {
        "schema": policy.config["materialization"]["cache_schema"],
        "experiment_id": policy.config["experiment_id"],
        "condition": condition,
        "entries": entries,
        "verified_frozen_artifacts": verified,
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
    }
    policy.validate_cache_manifest(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "status": "development_features_materialized",
        "experiment_id": policy.config["experiment_id"],
        "condition": condition,
        "quality_checks": quality_rows,
        "protected_bearings_not_read": ["Bearing3_4", "Bearing3_5"],
        "training_performed": False,
    }
    report_path = output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return manifest_path, report_path
