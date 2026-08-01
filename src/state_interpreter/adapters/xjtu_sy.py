"""Read the XJTU-SY run-to-failure bearing dataset.

The adapter preserves bearing-level chronology and returns raw two-channel
vibration measurements.  Signal transforms such as STFT deliberately live in
a later preprocessing layer.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from state_interpreter.contracts import RawMeasurement

from .interfaces import DatasetAdapter


_CONDITION_PATTERN = re.compile(
    r"^(?P<speed>\d+(?:\.\d+)?)Hz(?P<load>\d+(?:\.\d+)?)kN$"
)
_EXPECTED_HEADER = (
    "Horizontal_vibration_signals",
    "Vertical_vibration_signals",
)


def _natural_bearing_key(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"Bearing(\d+)_(\d+)", name)
    if match is None:
        raise ValueError(f"invalid XJTU-SY bearing directory name: {name}")
    return int(match.group(1)), int(match.group(2))


class XJTUSYDatasetAdapter(DatasetAdapter):
    """Yield ordered XJTU-SY measurements as ``RawMeasurement`` objects."""

    def __init__(
        self,
        root: str | Path,
        *,
        conditions: Sequence[str] | None = None,
        bearings: Sequence[str] | None = None,
        expected_samples: int = 32768,
        sampling_frequency_hz: float = 25600.0,
        measurement_interval_minutes: float = 1.0,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"XJTU-SY root does not exist: {self.root}")
        if expected_samples <= 0:
            raise ValueError("expected_samples must be positive")
        if sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        if measurement_interval_minutes <= 0:
            raise ValueError("measurement_interval_minutes must be positive")

        self.conditions = set(conditions) if conditions is not None else None
        self.bearings = set(bearings) if bearings is not None else None
        self.expected_samples = expected_samples
        self.sampling_frequency_hz = sampling_frequency_hz
        self.measurement_interval_minutes = measurement_interval_minutes

    def _selected_bearing_dirs(self) -> list[tuple[Path, float, float]]:
        selected: list[tuple[Path, float, float]] = []
        condition_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        for condition_dir in condition_dirs:
            match = _CONDITION_PATTERN.fullmatch(condition_dir.name)
            if match is None:
                continue
            if self.conditions is not None and condition_dir.name not in self.conditions:
                continue

            bearing_dirs = sorted(
                (path for path in condition_dir.iterdir() if path.is_dir()),
                key=lambda path: _natural_bearing_key(path.name),
            )
            for bearing_dir in bearing_dirs:
                if self.bearings is not None and bearing_dir.name not in self.bearings:
                    continue
                selected.append(
                    (
                        bearing_dir,
                        float(match.group("speed")),
                        float(match.group("load")),
                    )
                )

        if not selected:
            raise ValueError("no XJTU-SY bearing directories matched the selection")
        return selected

    @staticmethod
    def _ordered_csv_files(bearing_dir: Path) -> list[Path]:
        try:
            files = sorted(
                bearing_dir.glob("*.csv"),
                key=lambda path: int(path.stem),
            )
        except ValueError as exc:
            raise ValueError(
                f"CSV filenames must be integer measurement indices: {bearing_dir}"
            ) from exc
        if not files:
            raise ValueError(f"bearing directory contains no CSV files: {bearing_dir}")

        indices = [int(path.stem) for path in files]
        expected = list(range(1, indices[-1] + 1))
        if indices != expected:
            raise ValueError(
                f"measurement indices must be consecutive from 1 in {bearing_dir}"
            )
        return files

    def _read_vibration(self, path: Path) -> torch.Tensor:
        horizontal: list[float] = []
        vertical: list[float] = []
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise ValueError(f"empty XJTU-SY CSV: {path}") from exc
            if header != _EXPECTED_HEADER:
                raise ValueError(
                    f"unexpected CSV header in {path}: {header}; "
                    f"expected {_EXPECTED_HEADER}"
                )

            for row_number, row in enumerate(reader, start=2):
                if len(row) != 2:
                    raise ValueError(
                        f"expected 2 columns in {path} at row {row_number}, "
                        f"got {len(row)}"
                    )
                try:
                    horizontal_value = float(row[0])
                    vertical_value = float(row[1])
                except ValueError as exc:
                    raise ValueError(
                        f"non-numeric vibration value in {path} at row {row_number}"
                    ) from exc
                if not (
                    math.isfinite(horizontal_value)
                    and math.isfinite(vertical_value)
                ):
                    raise ValueError(
                        f"non-finite vibration value in {path} at row {row_number}"
                    )
                horizontal.append(horizontal_value)
                vertical.append(vertical_value)

        if len(horizontal) != self.expected_samples:
            raise ValueError(
                f"expected {self.expected_samples} samples in {path}, "
                f"got {len(horizontal)}"
            )
        return torch.tensor((horizontal, vertical), dtype=torch.float32)

    def iter_measurements(self) -> Iterable[RawMeasurement]:
        for bearing_dir, speed_hz, load_kn in self._selected_bearing_dirs():
            condition_name = bearing_dir.parent.name
            episode_id = f"{condition_name}/{bearing_dir.name}"
            for csv_path in self._ordered_csv_files(bearing_dir):
                measurement_number = int(csv_path.stem)
                step_id = measurement_number - 1
                yield RawMeasurement(
                    episode_id=episode_id,
                    step_id=step_id,
                    time_index=step_id * self.measurement_interval_minutes,
                    vibration=self._read_vibration(csv_path),
                    operating_conditions={
                        "rotational_frequency_hz": speed_hz,
                        "radial_load_kn": load_kn,
                    },
                    metadata={
                        "condition_id": condition_name,
                        "bearing_id": bearing_dir.name,
                        "measurement_number": measurement_number,
                        "sampling_frequency_hz": self.sampling_frequency_hz,
                        "time_unit": "minutes",
                        "source_path": str(csv_path),
                    },
                )

