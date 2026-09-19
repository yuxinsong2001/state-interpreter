import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_source_conditional_mmd_condition3_development import (  # noqa: E402
    ARMS, TOKEN, batch_tensors, model_new, train_windows, validate_config,
)
from state_interpreter.conditional_mmd_preflight import (
    median_squared_distance, paired_epoch_indices, paired_mmd2,
)


def config() -> dict:
    return json.loads((ROOT / "configs/xjtu_condition3_source_conditional_mmd_development_v1.json").read_text())


def test_protocol_preflight_without_cache_read():
    verified = validate_config(config())
    assert len(verified) == 7
    assert ARMS == ("source_only", "global_mmd", "conditional_mmd")
    assert TOKEN == "TRAIN_SOURCE_CONDITIONAL_MMD_CONDITION3_DEVELOPMENT_ONCE"


def test_protected_bearing_is_rejected():
    changed = copy.deepcopy(config())
    changed["development_bearings"][2] = "Bearing3_4"
    with pytest.raises(ValueError, match="development split"):
        validate_config(changed)


def test_weight_and_frozen_code_are_rejected_when_changed():
    changed = copy.deepcopy(config())
    changed["regularization"]["weight"] = 0.2
    with pytest.raises(ValueError, match="weight"):
        validate_config(changed)
    changed = copy.deepcopy(config())
    changed["frozen_artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen artifact"):
        validate_config(changed)


def test_model_initialization_and_paired_batch_are_reproducible():
    first = model_new(20260918)
    second = model_new(20260918)
    assert all(np.array_equal(a.detach().numpy(), b.detach().numpy())
               for a, b in zip(first.parameters(), second.parameters()))
    sequences = {"Bearing3_1": np.zeros((300, 65), dtype=np.float32),
                 "Bearing3_2": np.ones((371, 65), dtype=np.float32)}
    bearings = ["Bearing3_1", "Bearing3_2"]
    windows, lengths, positions = train_windows(sequences, bearings)
    endpoints, sources, bins = paired_epoch_indices((300, 371), seed=17)
    batch, valid, target, source, time_bin = batch_tensors(
        endpoints[0], sources[0], bins[0], sequences, bearings,
        windows, lengths, positions,
    )
    assert batch.shape == (60, 128, 65)
    assert valid.shape == target.shape == source.shape == time_bin.shape == (60,)
    assert int((source == 0).sum()) == int((source == 1).sum()) == 30


def test_no_training_without_exact_token():
    import subprocess
    output = ROOT / config()["output_directory"]

    def snapshot() -> dict:
        if not output.exists():
            return {}
        return {
            str(path.relative_to(output)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in output.rglob("*") if path.is_file()
        }

    before = snapshot()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_source_conditional_mmd_condition3_development.py"),
         "--config", str(ROOT / "configs/xjtu_condition3_source_conditional_mmd_development_v1.json"),
         "--confirm", "WRONG"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "exact development training token required" in result.stderr
    assert snapshot() == before


def test_three_arms_take_finite_synthetic_training_step():
    sequence_a = np.zeros((300, 65), dtype=np.float32)
    sequence_b = np.ones((371, 65), dtype=np.float32)
    sequence_a[:, 0] = np.linspace(0, 1, 300)
    sequence_b[:, 0] = np.linspace(0, 1, 371)
    sequences = {"Bearing3_1": sequence_a, "Bearing3_2": sequence_b}
    bearings = ["Bearing3_1", "Bearing3_2"]
    windows, lengths, positions = train_windows(sequences, bearings)
    endpoints, sources, bins = paired_epoch_indices((300, 371), seed=20260918)
    batch, valid, target, source, time_bin = batch_tensors(
        endpoints[0], sources[0], bins[0], sequences, bearings,
        windows, lengths, positions,
    )
    initial_parameters = None
    for arm in ARMS:
        model = model_new(20260918)
        if initial_parameters is None:
            initial_parameters = model.gru.weight_ih_l0.detach().clone()
        else:
            assert torch.equal(model.gru.weight_ih_l0, initial_parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        output = model(batch, lengths=valid, grl_coefficient=0.0)
        health = F.mse_loss(output.health.squeeze(-1), target)
        scale = median_squared_distance(output.embedding)
        global_loss, conditional_loss = paired_mmd2(output.embedding, source, time_bin, scale)
        penalty = {"source_only": health.new_zeros(()),
                   "global_mmd": global_loss,
                   "conditional_mmd": conditional_loss}[arm]
        loss = health + 0.1 * penalty
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.gru.weight_ih_l0.grad).all()
        optimizer.step()
