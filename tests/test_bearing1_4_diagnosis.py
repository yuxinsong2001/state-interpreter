from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = spec_from_file_location("diagnose_bearing1_4", SCRIPTS / "diagnose_bearing1_4.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_level_series_respects_calibration_length() -> None:
    latent = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    levels = MODULE.level_series(latent, calibration=5, window=3)
    assert isinstance(levels, np.ndarray)
    assert len(levels) == 15
