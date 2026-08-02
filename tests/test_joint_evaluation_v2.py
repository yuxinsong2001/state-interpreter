import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_joint_evaluation_v2.py"
SPEC = importlib.util.spec_from_file_location("run_joint_evaluation_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Config:
    arm_b_validation_bearing = "Bearing2_4"

    class split:
        blind_holdout = "Bearing2_5"


def test_rehearsal_can_only_select_validation_bearing() -> None:
    assert MODULE.resolve_evaluation_bearing(Config(), "rehearsal", None) == "Bearing2_4"


def test_blind_mode_rejects_missing_confirmation() -> None:
    with pytest.raises(ValueError, match="confirmation token"):
        MODULE.resolve_evaluation_bearing(Config(), "blind", None)


def test_blind_mode_requires_exact_confirmation() -> None:
    assert (
        MODULE.resolve_evaluation_bearing(
            Config(), "blind", MODULE.BLIND_CONFIRMATION
        )
        == "Bearing2_5"
    )


def test_comparison_uses_arm_b_minus_arm_a() -> None:
    arm_a = {"level_spearman_rho": 0.6, "reconstruction_mse_mean": 1.0}
    arm_b = {"level_spearman_rho": 0.8, "reconstruction_mse_mean": 0.5}
    result = MODULE.compare(arm_a, arm_b)
    assert result["arm_b_minus_arm_a_rho"] == pytest.approx(0.2)
