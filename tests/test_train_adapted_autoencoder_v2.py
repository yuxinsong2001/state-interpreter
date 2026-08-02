import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_adapted_autoencoder_v2.py"
)
SPEC = importlib.util.spec_from_file_location("train_adapted_autoencoder_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IdentityReconstructor(nn.Module):
    def forward(self, value):
        return type("Output", (), {"reconstruction": value})()


def test_evaluate_returns_sample_weighted_mse() -> None:
    tensors = torch.tensor([[[[1.0]]], [[[2.0]]]])
    loader = DataLoader(TensorDataset(tensors), batch_size=1, shuffle=False)
    loss = MODULE.evaluate(IdentityReconstructor(), loader, nn.MSELoss())
    assert loss == pytest.approx(0.0)


def test_evaluate_rejects_empty_loader() -> None:
    loader = DataLoader(TensorDataset(torch.empty(0, 1, 1, 1)), batch_size=1)
    with pytest.raises(ValueError, match="empty"):
        MODULE.evaluate(IdentityReconstructor(), loader, nn.MSELoss())

