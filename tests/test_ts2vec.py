import numpy as np
import torch

from state_interpreter.encoders.ts2vec import (
    TS2VecEncoder,
    hierarchical_contrastive_loss,
    sample_context_views,
)


def test_ts2vec_encoder_preserves_timestamp_axis() -> None:
    encoder = TS2VecEncoder(65, output_dims=32, hidden_dims=16, depth=3)
    encoder.eval()
    output = encoder(torch.randn(4, 64, 65))
    assert output.shape == (4, 64, 32)
    assert torch.isfinite(output).all()


def test_hierarchical_loss_backpropagates() -> None:
    encoder = TS2VecEncoder(65, output_dims=16, hidden_dims=8, depth=2)
    values = torch.randn(3, 32, 65)
    first = encoder(values)
    second = encoder(values)
    loss = hierarchical_contrastive_loss(first, second)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_context_views_align_to_same_crop_length_after_encoding_slice() -> None:
    values = torch.arange(2 * 40 * 3, dtype=torch.float32).reshape(2, 40, 3)
    first, second = sample_context_views(
        values, temporal_unit=0, random_state=np.random.default_rng(17)
    )
    assert first.shape[0] == second.shape[0] == 2
    assert first.shape[2] == second.shape[2] == 3
    assert first.shape[1] >= 2
    assert second.shape[1] >= 2


def test_encoder_handles_nan_padding() -> None:
    encoder = TS2VecEncoder(3, output_dims=8, hidden_dims=8, depth=2)
    encoder.eval()
    values = torch.randn(2, 20, 3)
    values[1, 15:] = float("nan")
    output = encoder(values)
    assert torch.isfinite(output).all()
