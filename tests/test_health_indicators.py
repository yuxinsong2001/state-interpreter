import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_health_indicators.py"
SPEC = importlib.util.spec_from_file_location("compare_health_indicators", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_self_relative_distance_uses_only_early_window() -> None:
    z = np.asarray([[0.0], [2.0], [4.0], [8.0]])
    result = MODULE.self_relative_distance(z, 0.5)
    assert result.tolist() == pytest.approx([1.0, 1.0, 3.0, 7.0])


def test_causal_moving_average_never_uses_future_values() -> None:
    result = MODULE.causal_moving_average(np.asarray([1.0, 3.0, 100.0]), 2)
    assert result.tolist() == pytest.approx([1.0, 2.0, 51.5])


def test_causal_slope_for_linear_sequence() -> None:
    result = MODULE.causal_slope(np.asarray([1.0, 3.0, 5.0, 7.0]), 3)
    assert result.tolist() == pytest.approx([0.0, 2.0, 2.0, 2.0])


def test_early_count_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        MODULE.early_count(10, 0.0)
