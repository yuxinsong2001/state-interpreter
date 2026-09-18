import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_source_dann_condition3_development",
    ROOT / "scripts" / "train_source_dann_condition3_development.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_right_padded_windows_are_causal() -> None:
    sequence = np.arange(18, dtype=np.float32).reshape(6, 3)
    windows, lengths = MODULE.right_padded_causal_windows(
        sequence,
        np.array([0, 2, 5]),
        context_length=4,
    )
    assert lengths.tolist() == [1, 3, 4]
    assert torch.equal(windows[0, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.count_nonzero(windows[0, 1:]) == 0
    assert torch.equal(windows[1, :3], torch.from_numpy(sequence[:3]))
    assert torch.equal(windows[2], torch.from_numpy(sequence[2:6]))


def test_wrong_execution_token_fails_before_data_access() -> None:
    config = {
        "status": "preregistered_preflight_no_cache_read",
        "execution_token": "EXACT",
    }
    with pytest.raises(PermissionError, match="exact Source-DANN"):
        MODULE.validate_execution_contract(config, "WRONG")


def test_right_padded_windows_reject_future_endpoint() -> None:
    with pytest.raises(ValueError, match="outside"):
        MODULE.right_padded_causal_windows(
            np.zeros((4, 2), dtype=np.float32),
            np.array([4]),
            context_length=3,
        )
