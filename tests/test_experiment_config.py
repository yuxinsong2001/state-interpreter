import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from state_interpreter.experiment_config import (
    load_cross_condition_config,
    verify_frozen_source_artifacts,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_config(checkpoint_hash: str, source_config_hash: str) -> dict:
    return {
        "config_id": "xjtu_cross_condition_v2_1",
        "status": "preregistered_not_run",
        "source_condition": "35Hz12kN",
        "target_condition": "37.5Hz11kN",
        "data_split": {
            "development_or_train_bearings": [
                "Bearing2_1",
                "Bearing2_2",
                "Bearing2_3",
            ],
            "validation_bearing": "Bearing2_4",
            "strict_blind_holdout": "Bearing2_5",
        },
        "blind_holdout_audit": {
            "signal_statistics_inspected": False,
            "latent_or_state_outputs_inspected": False,
        },
        "shared_interpreter": {
            "embedding_dim": 8,
            "calibration_steps": 15,
            "temporal_window": 10,
            "state_fields": ["level", "trend", "movement"],
            "parameters_may_be_tuned": False,
        },
        "arms": {
            "arm_a_direct_generalization": {
                "checkpoint": "artifacts/source.pt",
                "checkpoint_sha256": checkpoint_hash,
                "source_config": "artifacts/source.json",
                "source_config_sha256_at_amendment": source_config_hash,
                "normalization": "load_source_checkpoint_statistics",
                "training_allowed": False,
            },
            "arm_b_adapted_encoder": {
                "train_bearings": [
                    "Bearing2_1",
                    "Bearing2_2",
                    "Bearing2_3",
                ],
                "validation_bearing": "Bearing2_4",
                "normalization_fit_on": [
                    "Bearing2_1",
                    "Bearing2_2",
                    "Bearing2_3",
                ],
            },
        },
        "joint_blind_evaluation": {
            "single_script_reads_holdout_once": True,
            "both_arms_reported_together": True,
            "blind_holdout_evaluated": False,
            "retuning_after_evaluation_allowed": False,
        },
    }


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    artifacts = repository / "artifacts"
    artifacts.mkdir(parents=True)
    checkpoint = b"frozen checkpoint"
    source_config = b'{"locked": true}'
    (artifacts / "source.pt").write_bytes(checkpoint)
    (artifacts / "source.json").write_bytes(source_config)
    config = valid_config(sha256(checkpoint), sha256(source_config))
    config_path = repository / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return repository, config_path


def test_loads_immutable_two_arm_config(tmp_path: Path) -> None:
    _, path = write_fixture(tmp_path)
    config = load_cross_condition_config(path)

    assert config.bearings_for("arm_a_development") == (
        "Bearing2_1",
        "Bearing2_2",
        "Bearing2_3",
        "Bearing2_4",
    )
    assert config.bearings_for("arm_b_train") == (
        "Bearing2_1",
        "Bearing2_2",
        "Bearing2_3",
    )
    with pytest.raises(FrozenInstanceError):
        config.status = "completed"


def test_rejects_overlapping_split(tmp_path: Path) -> None:
    _, path = write_fixture(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["data_split"]["validation_bearing"] = "Bearing2_3"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="disjoint"):
        load_cross_condition_config(path)


def test_rejects_holdout_in_arm_b_training(tmp_path: Path) -> None:
    _, path = write_fixture(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["arms"]["arm_b_adapted_encoder"]["train_bearings"][-1] = "Bearing2_5"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must match the shared split"):
        load_cross_condition_config(path)


def test_rejects_non_train_normalization(tmp_path: Path) -> None:
    _, path = write_fixture(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["arms"]["arm_b_adapted_encoder"]["normalization_fit_on"] = [
        "Bearing2_1",
        "Bearing2_4",
    ]
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization"):
        load_cross_condition_config(path)


def test_rejects_previously_evaluated_blind_holdout(tmp_path: Path) -> None:
    _, path = write_fixture(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["joint_blind_evaluation"]["blind_holdout_evaluated"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="already been evaluated"):
        load_cross_condition_config(path)


def test_verifies_frozen_source_hashes(tmp_path: Path) -> None:
    repository, path = write_fixture(tmp_path)
    config = load_cross_condition_config(path)

    verified = verify_frozen_source_artifacts(config, repository)

    assert verified["source_checkpoint"] == config.source_checkpoint.sha256
    assert verified["source_config"] == config.source_config.sha256


def test_rejects_changed_frozen_source_artifact(tmp_path: Path) -> None:
    repository, path = write_fixture(tmp_path)
    config = load_cross_condition_config(path)
    (repository / "artifacts" / "source.pt").write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_frozen_source_artifacts(config, repository)

