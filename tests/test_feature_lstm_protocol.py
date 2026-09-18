import copy
import hashlib
import json
from pathlib import Path

import pytest

from state_interpreter.feature_lstm_protocol import (
    CACHE_SCHEMA,
    FeatureLSTMDevelopmentPolicy,
)


def config() -> dict:
    return {
        "experiment_id": "xjtu_condition3_feature_lstm_development_v1",
        "status": "preregistered_preflight_no_condition3_data_read",
        "condition": "40Hz10kN",
        "split": {
            "development": ["Bearing3_1", "Bearing3_2", "Bearing3_3"],
            "validation": ["Bearing3_4"],
            "blind": ["Bearing3_5"],
        },
        "outer_lobo_folds": [
            {"test": "Bearing3_1", "train": ["Bearing3_2", "Bearing3_3"]},
            {"test": "Bearing3_2", "train": ["Bearing3_1", "Bearing3_3"]},
            {"test": "Bearing3_3", "train": ["Bearing3_1", "Bearing3_2"]},
        ],
        "features": {
            "count": 65,
            "time_domain": 37,
            "frequency_domain": 28,
            "calibration_steps": 15,
            "window_size": 10,
        },
        "model": {
            "architecture": "FeatureLSTM",
            "n_features": 65,
            "hidden_size": 16,
            "dense_size": 16,
            "dropout": 0.2,
            "bidirectional": True,
            "pytorch_parameter_count": 11169,
        },
        "training": {
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
        },
        "materialization": {
            "cache_schema": CACHE_SCHEMA,
            "required_npz_fields": ["features", "step_ids", "source_paths"],
        },
        "tokens": {
            "materialize_development": "TOKEN",
        },
        "frozen_artifacts": [],
    }


def test_policy_authorizes_exact_development_set_only(tmp_path: Path) -> None:
    policy = FeatureLSTMDevelopmentPolicy(config(), tmp_path)
    assert policy.authorize_materialization(
        ["Bearing3_1", "Bearing3_2", "Bearing3_3"], "TOKEN"
    ) == ("Bearing3_1", "Bearing3_2", "Bearing3_3")
    with pytest.raises(ValueError, match="exactly"):
        policy.authorize_materialization(["Bearing3_1", "Bearing3_4"], "TOKEN")
    with pytest.raises(PermissionError, match="token"):
        policy.authorize_materialization(
            ["Bearing3_1", "Bearing3_2", "Bearing3_3"], "WRONG"
        )


@pytest.mark.parametrize("protected", ["Bearing3_4", "Bearing3_5"])
def test_policy_rejects_protected_bearing_in_development(
    tmp_path: Path, protected: str
) -> None:
    changed = config()
    changed["split"]["development"][-1] = protected
    with pytest.raises(ValueError, match="development split"):
        FeatureLSTMDevelopmentPolicy(changed, tmp_path)


def test_policy_rejects_changed_training_contract(tmp_path: Path) -> None:
    changed = config()
    changed["training"]["fixed_epochs"] = 51
    with pytest.raises(ValueError, match="training contract"):
        FeatureLSTMDevelopmentPolicy(changed, tmp_path)


def test_frozen_artifact_hash_is_verified(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.py"
    artifact.write_text("stable", encoding="utf-8")
    changed = config()
    changed["frozen_artifacts"] = [
        {
            "path": "artifact.py",
            "sha256": hashlib.sha256(b"stable").hexdigest(),
        }
    ]
    policy = FeatureLSTMDevelopmentPolicy(changed, tmp_path)
    assert policy.verify_frozen_artifacts() == {
        "artifact.py": hashlib.sha256(b"stable").hexdigest()
    }
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        policy.verify_frozen_artifacts()


def valid_manifest() -> dict:
    digest = "a" * 64
    return {
        "schema": CACHE_SCHEMA,
        "experiment_id": "xjtu_condition3_feature_lstm_development_v1",
        "condition": "40Hz10kN",
        "entries": [
            {
                "bearing_id": bearing,
                "sample_count": 20,
                "fields": ["features", "step_ids", "source_paths"],
                "path": f"{bearing}.npz",
                "sha256": digest,
            }
            for bearing in ("Bearing3_1", "Bearing3_2", "Bearing3_3")
        ],
    }


def test_cache_manifest_contract(tmp_path: Path) -> None:
    policy = FeatureLSTMDevelopmentPolicy(config(), tmp_path)
    policy.validate_cache_manifest(valid_manifest())

    protected = valid_manifest()
    protected["entries"][-1]["bearing_id"] = "Bearing3_5"
    with pytest.raises(ValueError, match="development bearings"):
        policy.validate_cache_manifest(protected)


def test_real_preregistration_loads_without_dataset_path() -> None:
    repository = Path(__file__).resolve().parents[1]
    path = repository / "configs/xjtu_condition3_feature_lstm_development_v1.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    policy = FeatureLSTMDevelopmentPolicy(copy.deepcopy(loaded), repository)
    assert policy.verify_frozen_artifacts()
