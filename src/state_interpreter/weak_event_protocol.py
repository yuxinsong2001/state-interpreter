"""Fail-closed, metadata-only preflight for Condition 3 weak-event development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from state_interpreter.features.bearing_features import FEATURE_NAMES


DEVELOPMENT = ("Bearing3_1", "Bearing3_2", "Bearing3_3")
PAIRS = {
    "rms_threshold_v1": ["h_rms", "v_rms"],
    "kurtosis_threshold_v1": ["h_kurtosis", "v_kurtosis"],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WeakEventDevelopmentPolicy:
    def __init__(self, config: dict, root: Path) -> None:
        self.config = config
        self.root = root.resolve()
        if config.get("experiment_id") != "xjtu_condition3_weak_event_development_v1":
            raise ValueError("wrong experiment")
        if config.get("status") != "preflight_only_not_executed":
            raise ValueError("wrong execution status")
        if config.get("condition") != "40Hz10kN":
            raise ValueError("wrong condition")
        if tuple(config.get("development_bearings", ())) != DEVELOPMENT:
            raise ValueError("development split changed")
        if config.get("protected_validation") != ["Bearing3_4"] or config.get("protected_blind") != ["Bearing3_5"]:
            raise ValueError("protected split changed")
        inp = config.get("input", {})
        if inp.get("cache_directory") != "cache/xjtu_condition3_feature_lstm_v1":
            raise ValueError("cache location changed")
        if inp.get("cache_schema") != "xjtu_feature_cache_v1" or inp.get("required_fields") != ["features", "step_ids", "source_paths"]:
            raise ValueError("cache schema changed")
        if inp.get("feature_count") != len(FEATURE_NAMES) or inp.get("feature_pairs") != PAIRS:
            raise ValueError("feature contract changed")
        entries = inp.get("cache_entries", [])
        if [entry.get("bearing_id") for entry in entries] != list(DEVELOPMENT):
            raise ValueError("cache bearing list changed")
        for bearing, entry in zip(DEVELOPMENT, entries):
            if entry.get("path") != f"{bearing}.npz" or entry.get("sample_count", 0) < 15:
                raise ValueError("cache entry contract changed")
            self._check_digest(entry.get("sha256"))
        self._check_digest(inp.get("manifest_sha256"))
        det = config.get("detector", {})
        if {key: det.get(key) for key in ("calibration_steps", "sigma_multiplier", "consecutive_exceedances", "min_reference_std")} != {
            "calibration_steps": 15, "sigma_multiplier": 2.0,
            "consecutive_exceedances": 5, "min_reference_std": 1e-8,
        }:
            raise ValueError("detector rule changed")
        if det.get("code_path") != "src/state_interpreter/weak_event_detector.py":
            raise ValueError("detector code path changed")
        self._check_digest(det.get("code_sha256"))
        if config.get("output_directory") != "results/2026-09-19_condition3_weak_event_development_v1":
            raise ValueError("output directory changed")
        if config.get("record_path") != "records/xjtu_condition3_weak_event_development_v1/evaluation_record.json":
            raise ValueError("record path changed")
        if config.get("interpretation") != "algorithmic candidate event only; no physical onset accuracy":
            raise ValueError("interpretation changed")

    @staticmethod
    def _check_digest(value: object) -> None:
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid sha256 digest")

    @classmethod
    def load(cls, config_path: Path, root: Path) -> "WeakEventDevelopmentPolicy":
        return cls(json.loads(config_path.read_text(encoding="utf-8")), root)

    def authorize(self, bearings: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        if tuple(bearings) != DEVELOPMENT:
            raise PermissionError("only exact development bearings are allowed")
        return DEVELOPMENT

    def verify_metadata_and_hashes(self) -> dict:
        """Hash cache bytes but never open NPZ arrays or read feature values."""
        inp = self.config["input"]
        cache = (self.root / inp["cache_directory"]).resolve()
        if not cache.is_relative_to(self.root):
            raise ValueError("cache escapes repository")
        manifest_path = cache / "manifest.json"
        if file_sha256(manifest_path) != inp["manifest_sha256"]:
            raise ValueError("cache manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != inp["cache_schema"] or manifest.get("condition") != self.config["condition"]:
            raise ValueError("cache manifest contract changed")
        if [x.get("bearing_id") for x in manifest.get("entries", [])] != list(DEVELOPMENT):
            raise ValueError("manifest includes unexpected bearing")
        verified: dict[str, str] = {}
        for expected, actual in zip(inp["cache_entries"], manifest["entries"]):
            if any(actual.get(k) != expected[k] for k in ("bearing_id", "path", "sha256", "sample_count")):
                raise ValueError("cache entry metadata changed")
            if actual.get("fields") != inp["required_fields"]:
                raise ValueError("cache fields changed")
            path = (cache / expected["path"]).resolve()
            if not path.is_relative_to(cache) or file_sha256(path) != expected["sha256"]:
                raise ValueError(f"cache payload hash mismatch: {expected['bearing_id']}")
            verified[expected["bearing_id"]] = expected["sha256"]
        code = (self.root / self.config["detector"]["code_path"]).resolve()
        if not code.is_relative_to(self.root) or file_sha256(code) != self.config["detector"]["code_sha256"]:
            raise ValueError("detector code hash mismatch")
        output = (self.root / self.config["output_directory"]).resolve()
        record = (self.root / self.config["record_path"]).resolve()
        if not output.is_relative_to(self.root) or not record.is_relative_to(self.root):
            raise ValueError("output escapes repository")
        if output.exists() or record.exists():
            raise FileExistsError("refusing to overwrite weak-event results or record")
        return {"status": "preflight_passed_no_npz_parsing_no_event_execution", "verified_cache_sha256": verified,
                "manifest_sha256": inp["manifest_sha256"], "detector_sha256": self.config["detector"]["code_sha256"],
                "development": list(DEVELOPMENT), "protected_not_read": ["Bearing3_4", "Bearing3_5"],
                "output_directory": self.config["output_directory"], "record_path": self.config["record_path"]}
