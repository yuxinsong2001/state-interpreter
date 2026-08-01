import csv

import pytest
import torch

from state_interpreter.adapters import XJTUSYDatasetAdapter


def write_measurement(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
        )
        writer.writerows(values)


def test_xjtu_adapter_preserves_numeric_time_order(tmp_path) -> None:
    bearing_dir = tmp_path / "35Hz12kN" / "Bearing1_1"
    write_measurement(bearing_dir / "2.csv", [(2.0, 20.0), (3.0, 30.0)])
    write_measurement(bearing_dir / "1.csv", [(1.0, 10.0), (1.5, 15.0)])
    adapter = XJTUSYDatasetAdapter(tmp_path, expected_samples=2)

    measurements = list(adapter.iter_measurements())

    assert [item.step_id for item in measurements] == [0, 1]
    assert [item.time_index for item in measurements] == [0.0, 1.0]
    assert measurements[0].episode_id == "35Hz12kN/Bearing1_1"
    assert measurements[0].vibration.shape == (2, 2)
    torch.testing.assert_close(
        measurements[0].vibration,
        torch.tensor([[1.0, 1.5], [10.0, 15.0]]),
    )
    assert measurements[0].operating_conditions == {
        "rotational_frequency_hz": 35.0,
        "radial_load_kn": 12.0,
    }


def test_xjtu_adapter_filters_condition_and_bearing(tmp_path) -> None:
    write_measurement(
        tmp_path / "35Hz12kN" / "Bearing1_1" / "1.csv",
        [(1.0, 2.0)],
    )
    write_measurement(
        tmp_path / "37.5Hz11kN" / "Bearing2_1" / "1.csv",
        [(3.0, 4.0)],
    )
    adapter = XJTUSYDatasetAdapter(
        tmp_path,
        conditions=["37.5Hz11kN"],
        bearings=["Bearing2_1"],
        expected_samples=1,
    )

    measurement = next(iter(adapter.iter_measurements()))

    assert measurement.episode_id == "37.5Hz11kN/Bearing2_1"
    assert measurement.operating_conditions["rotational_frequency_hz"] == 37.5


def test_xjtu_adapter_rejects_missing_measurement_index(tmp_path) -> None:
    bearing_dir = tmp_path / "35Hz12kN" / "Bearing1_1"
    write_measurement(bearing_dir / "1.csv", [(1.0, 2.0)])
    write_measurement(bearing_dir / "3.csv", [(3.0, 4.0)])
    adapter = XJTUSYDatasetAdapter(tmp_path, expected_samples=1)

    with pytest.raises(ValueError, match="consecutive"):
        list(adapter.iter_measurements())


def test_xjtu_adapter_rejects_non_finite_values(tmp_path) -> None:
    bearing_dir = tmp_path / "35Hz12kN" / "Bearing1_1"
    write_measurement(bearing_dir / "1.csv", [(float("nan"), 2.0)])
    adapter = XJTUSYDatasetAdapter(tmp_path, expected_samples=1)

    with pytest.raises(ValueError, match="non-finite"):
        list(adapter.iter_measurements())

