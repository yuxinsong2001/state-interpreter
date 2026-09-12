from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stability_study.py"
SPEC = spec_from_file_location("run_stability_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rotating_folds_are_disjoint_and_cover_every_test_bearing() -> None:
    bearings = tuple(f"Bearing1_{index}" for index in range(1, 6))
    folds = MODULE.make_folds(bearings)
    assert tuple(fold.test for fold in folds) == bearings
    for fold in folds:
        assert len(fold.train) == 3
        assert set(fold.train).isdisjoint({fold.validation, fold.test})


def test_confidence_interval_reports_mean_and_sample_count() -> None:
    result = MODULE.confidence_interval([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_safe_spearman_handles_constant_input() -> None:
    assert np.isnan(MODULE.safe_spearman(np.ones(3), np.arange(3)))
