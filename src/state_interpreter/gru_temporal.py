"""Small causal GRU for predictive latent-state interpretation."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class GRUSequenceOutput(NamedTuple):
    """Causal hidden states and one-step latent predictions."""

    hidden_sequence: torch.Tensor
    next_embedding_prediction: torch.Tensor
    final_hidden: torch.Tensor


class PredictiveGRUStateInterpreter(nn.Module):
    """Learn temporal latent dynamics through one-step prediction.

    At time ``t`` the hidden state and prediction depend only on embeddings up
    to and including ``t``. The prediction head estimates the embedding at
    ``t + 1`` during training; normalized lifetime is never a training target.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int = 8,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
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

    def forward(self, sequence: torch.Tensor) -> GRUSequenceOutput:
        if sequence.ndim != 3:
            raise ValueError(
                "sequence must have shape [batch, time, embedding_dim], "
                f"got {tuple(sequence.shape)}"
            )
        if sequence.shape[1] < 1:
            raise ValueError("sequence must contain at least one time step")
        if sequence.shape[2] != self.embedding_dim:
            raise ValueError(
                f"expected embedding_dim={self.embedding_dim}, "
                f"got {sequence.shape[2]}"
            )
        if not torch.is_floating_point(sequence):
            raise ValueError("sequence must be floating-point")
        if not bool(torch.isfinite(sequence).all()):
            raise ValueError("sequence contains NaN or infinite values")
        hidden_sequence, final_hidden = self.gru(sequence)
        prediction = self.prediction_head(hidden_sequence)
        return GRUSequenceOutput(
            hidden_sequence=hidden_sequence,
            next_embedding_prediction=prediction,
            final_hidden=final_hidden,
        )
