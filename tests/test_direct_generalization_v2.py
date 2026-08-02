import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_direct_generalization_v2.py"
)
SPEC = importlib.util.spec_from_file_location("run_direct_generalization_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def ready_rows() -> list[dict[str, object]]:
    return [
        {
            "phase": "calibrating",
            "normalized_lifetime": 0.0,
            "step_id": 0,
            "reconstruction_mse": 1.0,
            "level": "",
            "trend": "",
            "movement": "",
        },
        {
            "phase": "ready",
            "normalized_lifetime": 0.5,
            "step_id": 1,
            "reconstruction_mse": 2.0,
            "level": 1.0,
            "trend": 0.0,
            "movement": 0.5,
        },
        {
            "phase": "ready",
            "normalized_lifetime": 1.0,
            "step_id": 2,
            "reconstruction_mse": 3.0,
            "level": 2.0,
            "trend": 1.0,
            "movement": 1.5,
        },
    ]


def test_summarize_ready_states() -> None:
    result = MODULE.summarize_ready_states(ready_rows())

    assert result["total_samples"] == 3
    assert result["calibration_samples"] == 1
    assert result["ready_samples"] == 2
    assert result["level_spearman_rho"] == pytest.approx(1.0)
    assert result["movement_max_step"] == 2
    assert result["reconstruction_mse_mean"] == pytest.approx(2.0)


def test_summarize_requires_two_ready_states() -> None:
    with pytest.raises(ValueError, match="two READY"):
        MODULE.summarize_ready_states(ready_rows()[:2])


def test_roughness_smoothness_is_one_for_linear_sequence() -> None:
    import numpy as np

    assert MODULE.roughness_smoothness(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(
        1.0
    )

