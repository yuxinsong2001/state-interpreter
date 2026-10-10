"""GRU-only relative-state model with explicit prediction and level heads.

Level is a scaled relative-deviation proxy, NOT physical damage or health.
Trend and delayed prediction residual are computed outside the model.
"""
from typing import NamedTuple
import torch
from torch import nn
from .residual_gru import ResidualGRUStateInterpreter


class GRUStateReadoutOutput(NamedTuple):
    hidden_sequence: torch.Tensor
    next_embedding_prediction: torch.Tensor
    scaled_relative_level: torch.Tensor
    final_hidden: torch.Tensor


class GRUStateReadoutInterpreter(nn.Module):
    """Shared causal GRU with a residual forecast and nonnegative level head.

    Input: [batch, time, embedding_dim], centered and train-standardized.
    Level: [batch, time, 1], in TRAIN-target-scale units. The runner must
    restore target scale for reporting; no statistics are fitted here.
    """
    def __init__(self, *, embedding_dim: int, hidden_dim: int = 8, num_layers: int = 1):
        super().__init__()
        self.core = ResidualGRUStateInterpreter(embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, num_layers=num_layers)
        self.level_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, sequence: torch.Tensor) -> GRUStateReadoutOutput:
        value = self.core(sequence)
        level = self.level_head(value.hidden_sequence)
        return GRUStateReadoutOutput(value.hidden_sequence,
            value.next_embedding_prediction, level, value.final_hidden)
