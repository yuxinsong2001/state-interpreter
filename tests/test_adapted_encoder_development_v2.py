import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_adapted_encoder_development_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_adapted_encoder_development_v2", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compare_summaries_computes_preregistered_delta() -> None:
    arm_a = {
        "Bearing2_1": {
            "level_spearman_rho": 0.4,
            "reconstruction_mse_mean": 1.2,
        }
    }
    arm_b = [
        {
            "bearing_id": "Bearing2_1",
            "level_spearman_rho": 0.7,
            "reconstruction_mse_mean": 0.2,
        }
    ]
    result = MODULE.compare_summaries(arm_a, arm_b)
    assert result[0]["arm_b_minus_arm_a_rho"] == pytest.approx(0.3)
    assert result[0]["arm_b_reconstruction_mse_mean"] == pytest.approx(0.2)


def test_compare_summaries_rejects_missing_arm_a_bearing() -> None:
    with pytest.raises(ValueError, match="missing bearing"):
        MODULE.compare_summaries(
            {},
            [
                {
                    "bearing_id": "Bearing2_1",
                    "level_spearman_rho": 0.7,
                    "reconstruction_mse_mean": 0.2,
                }
            ],
        )
