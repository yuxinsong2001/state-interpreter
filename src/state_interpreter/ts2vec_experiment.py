"""Data-contract utilities for protected compact TS2Vec experiments."""

from __future__ import annotations

import numpy as np
import torch


def evenly_spaced_chunks(
    sequence: np.ndarray | torch.Tensor,
    *,
    chunk_length: int,
    chunk_count: int,
) -> torch.Tensor:
    values = torch.as_tensor(sequence, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("sequence must have shape [timestamps, features]")
    if chunk_length < 2 or chunk_length > values.shape[0]:
        raise ValueError("chunk_length must fit the sequence")
    if chunk_count < 1:
        raise ValueError("chunk_count must be positive")
    maximum_start = values.shape[0] - chunk_length
    starts = np.rint(np.linspace(0, maximum_start, chunk_count)).astype(np.int64)
    return torch.stack(
        [values[start : start + chunk_length] for start in starts.tolist()]
    )


def causal_context_windows(
    sequence: np.ndarray | torch.Tensor,
    endpoints: np.ndarray | torch.Tensor,
    *,
    context_length: int,
) -> torch.Tensor:
    """Return left-only contexts ending at each requested timestamp."""

    values = torch.as_tensor(sequence, dtype=torch.float32)
    indices = torch.as_tensor(endpoints, dtype=torch.long)
    if values.ndim != 2:
        raise ValueError("sequence must have shape [timestamps, features]")
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("endpoints must be a non-empty vector")
    if context_length < 2:
        raise ValueError("context_length must be at least two")
    if torch.any(indices < 0) or torch.any(indices >= values.shape[0]):
        raise ValueError("endpoint outside sequence")
    output = values.new_full(
        (indices.numel(), context_length, values.shape[1]), float("nan")
    )
    for row, endpoint in enumerate(indices.tolist()):
        start = max(0, endpoint - context_length + 1)
        context = values[start : endpoint + 1]
        output[row, -context.shape[0] :] = context
    return output
