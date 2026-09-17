"""Persistence-centred residual GRU for one-step latent prediction."""

from __future__ import annotations

import torch
from torch import nn

from .gru_temporal import GRUSequenceOutput


class ResidualGRUStateInterpreter(nn.Module):
    """Predict ``z[t+1] = z[t] + delta[t]`` with a causal GRU.

    The delta head is zero-initialised by default, so the untrained model is
    exactly the persistence forecast while retaining the same trainable
    capacity as the direct predictor.
    """

    def __init__(self, *, embedding_dim: int, hidden_dim: int = 8, num_layers: int = 1) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("embedding_dim, hidden_dim and num_layers must be positive")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.prediction_head = nn.Linear(hidden_dim, embedding_dim)
        nn.init.zeros_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)

    def forward(self, sequence: torch.Tensor) -> GRUSequenceOutput:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, embedding_dim]")
        if sequence.shape[1] < 1 or sequence.shape[2] != self.embedding_dim:
            raise ValueError("invalid time or embedding dimension")
        if not torch.is_floating_point(sequence) or not torch.isfinite(sequence).all():
            raise ValueError("finite floating sequence required")
        hidden_sequence, final_hidden = self.gru(sequence)
        prediction = sequence + self.prediction_head(hidden_sequence)
        return GRUSequenceOutput(hidden_sequence, prediction, final_hidden)
