import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_locked_holdout.py"
SPEC = importlib.util.spec_from_file_location("evaluate_locked_holdout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_config() -> dict[str, object]:
    return {
        "embedding_dim": 8,
        "calibration_steps": 15,
        "temporal_window": 10,
        "holdout": {
            "bearing": "Bearing1_5",
            "evaluated_during_selection": False,
        },
    }


def test_load_locked_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config()), encoding="utf-8")
    assert MODULE.load_locked_config(path)["calibration_steps"] == 15


def test_load_locked_config_rejects_previously_evaluated(tmp_path: Path) -> None:
    config = valid_config()
    config["final_evaluation"] = {"evaluated": True}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="already"):
        MODULE.load_locked_config(path)


def test_load_locked_config_rejects_selection_leakage(tmp_path: Path) -> None:
    config = valid_config()
    config["holdout"]["evaluated_during_selection"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="not isolated"):
        MODULE.load_locked_config(path)


def test_select_holdout_rows_returns_only_designated_bearing() -> None:
    rows = [
        {"bearing_id": "Bearing1_4", "step_id": "0"},
        {"bearing_id": "Bearing1_5", "step_id": "0"},
        {"bearing_id": "Bearing1_5", "step_id": "1"},
    ]
    selected = MODULE.select_holdout_rows(rows, "Bearing1_5")
    assert [row["bearing_id"] for row in selected] == ["Bearing1_5", "Bearing1_5"]


def test_select_holdout_rows_rejects_non_increasing_steps() -> None:
    rows = [
        {"bearing_id": "Bearing1_5", "step_id": "1"},
        {"bearing_id": "Bearing1_5", "step_id": "1"},
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        MODULE.select_holdout_rows(rows, "Bearing1_5")
