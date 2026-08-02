import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_interpreter_parameters.py"
SPEC = importlib.util.spec_from_file_location("sweep_interpreter_parameters", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_grid() -> None:
    assert MODULE.parse_grid("5,10, 15") == (5, 10, 15)


@pytest.mark.parametrize("invalid", ["", "0,5", "5,5"])
def test_parse_grid_rejects_invalid_values(invalid: str) -> None:
    with pytest.raises(ValueError):
        MODULE.parse_grid(invalid)


def test_development_selection_excludes_holdout() -> None:
    rows = [
        {"bearing_id": "Bearing1_1"},
        {"bearing_id": "Bearing1_2"},
        {"bearing_id": "Bearing1_3"},
        {"bearing_id": "Bearing1_4"},
        {"bearing_id": "Bearing1_5"},
    ]
    selected, excluded = MODULE.select_development_rows(
        rows,
        ("Bearing1_1", "Bearing1_2", "Bearing1_3"),
        "Bearing1_4",
    )
    assert [row["bearing_id"] for row in selected] == [
        "Bearing1_1",
        "Bearing1_2",
        "Bearing1_3",
        "Bearing1_4",
    ]
    assert excluded == ["Bearing1_5"]


def test_development_selection_rejects_missing_validation() -> None:
    rows = [
        {"bearing_id": "Bearing1_1"},
        {"bearing_id": "Bearing1_2"},
        {"bearing_id": "Bearing1_3"},
    ]
    with pytest.raises(ValueError, match="missing"):
        MODULE.select_development_rows(
            rows,
            ("Bearing1_1", "Bearing1_2", "Bearing1_3"),
            "Bearing1_4",
        )
