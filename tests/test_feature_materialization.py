import json
from pathlib import Path

import numpy as np
import pytest

from state_interpreter.feature_lstm_protocol import FeatureLSTMDevelopmentPolicy
from state_interpreter.feature_materialization import materialize_development_cache


def _config() -> dict:
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
            "cache_schema": "xjtu_feature_cache_v1",
            "required_npz_fields": ["features", "step_ids", "source_paths"],
            "cache_directory": "cache/test",
        },
        "tokens": {"materialize_development": "TOKEN"},
        "frozen_artifacts": [],
    }


def _write_bearing(root: Path, bearing: str) -> None:
    directory = root / "40Hz10kN" / bearing
    directory.mkdir(parents=True)
    x = np.linspace(0, 2 * np.pi, 32)
    for index in range(1, 16):
        values = np.column_stack((np.sin(x) + index, np.cos(x) - index))
        np.savetxt(
            directory / f"{index}.csv",
            values,
            delimiter=",",
            header="Horizontal_vibration_signals,Vertical_vibration_signals",
            comments="",
        )


def test_wrong_token_fails_before_dataset_access(tmp_path: Path) -> None:
    policy = FeatureLSTMDevelopmentPolicy(_config(), tmp_path)
    with pytest.raises(PermissionError, match="token"):
        materialize_development_cache(
            policy=policy,
            dataset_root=tmp_path / "missing-dataset",
            token="WRONG",
            output_dir=tmp_path / "cache/test",
            expected_samples=32,
        )


def test_materializes_exact_development_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for bearing in ("Bearing3_1", "Bearing3_2", "Bearing3_3"):
        _write_bearing(dataset, bearing)
    policy = FeatureLSTMDevelopmentPolicy(_config(), tmp_path)
    output = tmp_path / "cache/test"
    manifest_path, report_path = materialize_development_cache(
        policy=policy,
        dataset_root=dataset,
        token="TOKEN",
        output_dir=output,
        expected_samples=32,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [row["bearing_id"] for row in manifest["entries"]] == [
        "Bearing3_1",
        "Bearing3_2",
        "Bearing3_3",
    ]
    assert manifest["protected_bearings_not_read"] == ["Bearing3_4", "Bearing3_5"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["training_performed"] is False
    with np.load(output / "Bearing3_1.npz") as payload:
        assert payload["features"].shape == (15, 65)
        assert payload["step_ids"].tolist() == list(range(15))
        assert payload["source_paths"][0] == "40Hz10kN/Bearing3_1/1.csv"

    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_development_cache(
            policy=policy,
            dataset_root=dataset,
            token="TOKEN",
            output_dir=output,
            expected_samples=32,
        )
