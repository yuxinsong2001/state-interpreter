import numpy as np
import json
from pathlib import Path
import sys
import pytest
import torch

from state_interpreter.conditional_mmd_preflight import (
    median_squared_distance, paired_epoch_indices, paired_mmd2, rbf_mmd2,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preflight_source_conditional_mmd_numerics import validate_scope


def test_paired_epoch_is_balanced_deterministic_and_in_range():
    first = paired_epoch_indices((2538, 371), seed=123)
    second = paired_epoch_indices((2538, 371), seed=123)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    endpoints, sources, bins = first
    assert endpoints.shape == sources.shape == bins.shape == (10, 60)
    for batch in range(10):
        for source in range(2):
            for time_bin in range(5):
                mask = (sources[batch] == source) & (bins[batch] == time_bin)
                assert mask.sum() == 6
                assert np.all(endpoints[batch, mask] < (2538, 371)[source])
    assert np.unique(endpoints[sources == 1]).size == 299


def test_mmd_identity_symmetry_and_finite_gradient():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]], requires_grad=True)
    y = torch.tensor([[2.0, 2.0], [3.0, 2.0], [4.0, 3.0]], requires_grad=True)
    scale = median_squared_distance(torch.cat((x, y)))
    assert abs(rbf_mmd2(x, x, scale).detach().item()) < 1e-6
    xy = rbf_mmd2(x, y, scale)
    assert xy.item() == pytest.approx(rbf_mmd2(y, x, scale).item(), abs=1e-6)
    assert xy.item() > 0
    xy.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_paired_mmd_rejects_missing_bin():
    z = torch.randn(10, 4)
    source = torch.tensor([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="five bins"):
        paired_mmd2(z, source, torch.zeros(10, dtype=torch.long), torch.tensor(1.0))


def test_sampler_rejects_protected_or_short_lengths():
    with pytest.raises(ValueError):
        paired_epoch_indices((2538, 299), seed=1)
    with pytest.raises(ValueError):
        paired_epoch_indices((2538, 371, 2496), seed=1)


def test_preflight_scope_fails_closed():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/xjtu_condition3_source_dann_development_v1.json").read_text())
    with pytest.raises(PermissionError):
        validate_scope(config, "WRONG")
    with pytest.raises(ValueError, match="protected"):
        changed = json.loads(json.dumps(config))
        changed["protected_validation"] = ["Bearing3_4", "Bearing3_5"]
        validate_scope(changed, "PREFLIGHT_SOURCE_CONDITIONAL_MMD_DEVELOPMENT_ONLY")
