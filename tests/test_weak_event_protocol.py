"""Protocol tests use synthetic metadata only, never actual bearing payloads."""

from copy import deepcopy
import hashlib
import json

import pytest

from state_interpreter.weak_event_protocol import WeakEventDevelopmentPolicy


def fixture(tmp_path):
    root = tmp_path
    cache = root / "cache/xjtu_condition3_feature_lstm_v1"
    cache.mkdir(parents=True)
    entries = []
    for bearing in ("Bearing3_1", "Bearing3_2", "Bearing3_3"):
        data = f"synthetic-{bearing}".encode()
        (cache / f"{bearing}.npz").write_bytes(data)
        entries.append({"bearing_id": bearing, "path": f"{bearing}.npz", "sample_count": 20,
                        "sha256": hashlib.sha256(data).hexdigest(), "fields": ["features", "step_ids", "source_paths"]})
    manifest = {"schema": "xjtu_feature_cache_v1", "condition": "40Hz10kN", "entries": entries}
    manifest_bytes = json.dumps(manifest).encode()
    (cache / "manifest.json").write_bytes(manifest_bytes)
    code = root / "src/state_interpreter/weak_event_detector.py"
    code.parent.mkdir(parents=True)
    code.write_text("synthetic code", encoding="utf-8")
    config = {
        "experiment_id": "xjtu_condition3_weak_event_development_v1", "status": "preflight_only_not_executed",
        "condition": "40Hz10kN", "development_bearings": ["Bearing3_1", "Bearing3_2", "Bearing3_3"],
        "protected_validation": ["Bearing3_4"], "protected_blind": ["Bearing3_5"],
        "input": {"cache_directory": "cache/xjtu_condition3_feature_lstm_v1",
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "cache_schema": "xjtu_feature_cache_v1",
                  "required_fields": ["features", "step_ids", "source_paths"], "feature_count": 65,
                  "feature_pairs": {"rms_threshold_v1": ["h_rms", "v_rms"],
                                    "kurtosis_threshold_v1": ["h_kurtosis", "v_kurtosis"]},
                  "cache_entries": [{k: x[k] for k in ("bearing_id", "path", "sha256", "sample_count")} for x in entries]},
        "detector": {"calibration_steps": 15, "sigma_multiplier": 2.0, "consecutive_exceedances": 5,
                     "min_reference_std": 1e-8, "code_path": "src/state_interpreter/weak_event_detector.py",
                     "code_sha256": hashlib.sha256(code.read_bytes()).hexdigest()},
        "output_directory": "results/2026-09-19_condition3_weak_event_development_v1",
        "record_path": "records/xjtu_condition3_weak_event_development_v1/evaluation_record.json",
        "interpretation": "algorithmic candidate event only; no physical onset accuracy",
    }
    return root, config


def test_preflight_accepts_exact_synthetic_manifest(tmp_path):
    root, config = fixture(tmp_path)
    policy = WeakEventDevelopmentPolicy(config, root)
    assert policy.verify_metadata_and_hashes()["status"] == "preflight_passed_no_npz_parsing_no_event_execution"


def test_protected_bearing_rejected_before_payload_access(tmp_path):
    root, config = fixture(tmp_path)
    policy = WeakEventDevelopmentPolicy(config, root)
    with pytest.raises(PermissionError):
        policy.authorize(["Bearing3_1", "Bearing3_4"])
    changed = deepcopy(config)
    changed["development_bearings"][2] = "Bearing3_5"
    with pytest.raises(ValueError):
        WeakEventDevelopmentPolicy(changed, root)


def test_fails_closed_on_tampering_and_existing_output(tmp_path):
    root, config = fixture(tmp_path)
    cache_file = root / config["input"]["cache_directory"] / "Bearing3_2.npz"
    cache_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload hash"):
        WeakEventDevelopmentPolicy(config, root).verify_metadata_and_hashes()
    root, config = fixture(tmp_path / "second")
    (root / config["output_directory"]).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        WeakEventDevelopmentPolicy(config, root).verify_metadata_and_hashes()


def test_rule_change_rejected(tmp_path):
    root, config = fixture(tmp_path)
    changed = deepcopy(config)
    changed["detector"]["consecutive_exceedances"] = 4
    with pytest.raises(ValueError, match="rule"):
        WeakEventDevelopmentPolicy(changed, root)
