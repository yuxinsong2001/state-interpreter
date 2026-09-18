"""Fail-closed protocol for the Condition 3 Feature LSTM development study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_DEVELOPMENT = ("Bearing3_1", "Bearing3_2", "Bearing3_3")
EXPECTED_VALIDATION = ("Bearing3_4",)
EXPECTED_BLIND = ("Bearing3_5",)
EXPECTED_FOLDS = (
    ("Bearing3_1", ("Bearing3_2", "Bearing3_3")),
    ("Bearing3_2", ("Bearing3_1", "Bearing3_3")),
    ("Bearing3_3", ("Bearing3_1", "Bearing3_2")),
)
CACHE_SCHEMA = "xjtu_feature_cache_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureLSTMDevelopmentPolicy:
    """Validate preregistration and authorize development-only data access."""

    def __init__(self, config: dict[str, Any], repository_root: Path) -> None:
        self.config = config
        self.repository_root = repository_root.resolve()
        self._validate_config()

    @classmethod
    def load(
        cls, config_path: str | Path, repository_root: str | Path
    ) -> "FeatureLSTMDevelopmentPolicy":
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(config, Path(repository_root))

    def _validate_config(self) -> None:
        if self.config.get("status") != "preregistered_preflight_no_condition3_data_read":
            raise ValueError("invalid Feature LSTM preregistration status")
        if self.config.get("condition") != "40Hz10kN":
            raise ValueError("condition must remain 40Hz10kN")
        split = self.config.get("split", {})
        if tuple(split.get("development", ())) != EXPECTED_DEVELOPMENT:
            raise ValueError("development split changed")
        if tuple(split.get("validation", ())) != EXPECTED_VALIDATION:
            raise ValueError("validation bearing changed")
        if tuple(split.get("blind", ())) != EXPECTED_BLIND:
            raise ValueError("blind bearing changed")
        folds = tuple(
            (fold.get("test"), tuple(fold.get("train", ())))
            for fold in self.config.get("outer_lobo_folds", ())
        )
        if folds != EXPECTED_FOLDS:
            raise ValueError("outer LOBO folds changed")
        feature = self.config.get("features", {})
        if feature != {
            "count": 65,
            "time_domain": 37,
            "frequency_domain": 28,
            "calibration_steps": 15,
            "window_size": 10,
        }:
            raise ValueError("feature contract changed")
        model = self.config.get("model", {})
        if model != {
            "architecture": "FeatureLSTM",
            "n_features": 65,
            "hidden_size": 16,
            "dense_size": 16,
            "dropout": 0.2,
            "bidirectional": True,
            "pytorch_parameter_count": 11169,
        }:
            raise ValueError("model contract changed")
        training = self.config.get("training", {})
        required_training = {
            "target": "normalized_remaining_useful_life",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": "HuberLoss",
            "huber_delta": 0.08,
            "batch_size": 32,
            "fixed_epochs": 50,
            "checkpoint_selection": "fixed_final_epoch_no_outer_test_selection",
            "seeds": [20260918, 20260921, 20260924],
        }
        if training != required_training:
            raise ValueError("training contract changed")
        materialization = self.config.get("materialization", {})
        if materialization.get("cache_schema") != CACHE_SCHEMA:
            raise ValueError("cache schema changed")
        if materialization.get("required_npz_fields") != [
            "features",
            "step_ids",
            "source_paths",
        ]:
            raise ValueError("cache field contract changed")

    def verify_frozen_artifacts(self) -> dict[str, str]:
        verified: dict[str, str] = {}
        for artifact in self.config.get("frozen_artifacts", []):
            relative_path = Path(artifact["path"])
            path = (self.repository_root / relative_path).resolve()
            try:
                path.relative_to(self.repository_root)
            except ValueError as error:
                raise ValueError("frozen artifact escapes repository") from error
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                raise ValueError(f"frozen artifact mismatch: {relative_path.as_posix()}")
            verified[relative_path.as_posix()] = artifact["sha256"]
        return verified

    def authorize_materialization(
        self, bearings: list[str] | tuple[str, ...], token: str
    ) -> tuple[str, ...]:
        requested = tuple(bearings)
        if requested != EXPECTED_DEVELOPMENT:
            raise ValueError(
                f"materialization may read exactly {EXPECTED_DEVELOPMENT}"
            )
        expected_token = self.config["tokens"]["materialize_development"]
        if token != expected_token:
            raise PermissionError("exact materialization token required")
        return requested

    def validate_cache_manifest(self, manifest: dict[str, Any]) -> None:
        """Validate a future cache manifest without opening any NPZ payload."""

        if manifest.get("schema") != CACHE_SCHEMA:
            raise ValueError("invalid cache manifest schema")
        if manifest.get("experiment_id") != self.config.get("experiment_id"):
            raise ValueError("cache experiment_id mismatch")
        if manifest.get("condition") != self.config.get("condition"):
            raise ValueError("cache condition mismatch")
        entries = manifest.get("entries", [])
        if tuple(entry.get("bearing_id") for entry in entries) != EXPECTED_DEVELOPMENT:
            raise ValueError("cache must contain exactly the development bearings")
        required_fields = tuple(self.config["materialization"]["required_npz_fields"])
        for entry in entries:
            if entry.get("sample_count", 0) <= 0:
                raise ValueError("cache sample_count must be positive")
            if tuple(entry.get("fields", ())) != required_fields:
                raise ValueError("cache entry fields changed")
            if not str(entry.get("path", "")).endswith(".npz"):
                raise ValueError("cache entry path must end with .npz")
            digest = str(entry.get("sha256", ""))
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("cache entry sha256 is invalid")
