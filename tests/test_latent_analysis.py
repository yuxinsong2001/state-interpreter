import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_latent_space.py"
SPEC = importlib.util.spec_from_file_location("analyze_latent_space", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_fraction_mask_keeps_episodes_separate() -> None:
    mask = MODULE.first_fraction_mask(
        ["a", "a", "a", "b", "b"], [0, 1, 2, 0, 1], 0.34
    )
    assert mask.tolist() == [True, True, False, True, False]


def test_consecutive_displacement_resets_at_episode_boundary() -> None:
    z = np.asarray([[0.0], [3.0], [10.0], [14.0]])
    result = MODULE.consecutive_displacement(z, ["a", "a", "b", "b"], [0, 1, 0, 1])
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(3.0)
    assert np.isnan(result[2])
    assert result[3] == pytest.approx(4.0)


def test_consecutive_displacement_rejects_non_increasing_steps() -> None:
    with pytest.raises(ValueError, match="not increasing"):
        MODULE.consecutive_displacement(
            np.asarray([[0.0], [1.0]]), ["a", "a"], [1, 1]
        )
