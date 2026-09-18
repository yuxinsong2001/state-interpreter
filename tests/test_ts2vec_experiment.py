import numpy as np
import torch

from state_interpreter.ts2vec_experiment import (
    causal_context_windows,
    evenly_spaced_chunks,
)


def test_even_chunks_cover_both_sequence_ends() -> None:
    sequence = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    chunks = evenly_spaced_chunks(sequence, chunk_length=8, chunk_count=3)
    assert chunks.shape == (3, 8, 1)
    assert chunks[0, 0, 0] == 0
    assert chunks[-1, -1, 0] == 19


def test_causal_context_has_no_future_and_left_padding() -> None:
    sequence = np.arange(10, dtype=np.float32)[:, None]
    contexts = causal_context_windows(sequence, [0, 3, 9], context_length=4)
    assert contexts.shape == (3, 4, 1)
    assert torch.isnan(contexts[0, :3]).all()
    assert contexts[0, -1, 0] == 0
    assert contexts[1, :, 0].tolist() == [0, 1, 2, 3]
    assert contexts[2, :, 0].tolist() == [6, 7, 8, 9]


def test_invalid_endpoint_is_rejected() -> None:
    sequence = torch.zeros(5, 2)
    try:
        causal_context_windows(sequence, [5], context_length=3)
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("expected invalid endpoint failure")
