"""Minimal State Interpreter interfaces.

The module intentionally does not depend on a VibFM checkpoint. It accepts an
already extracted ``z_health`` tensor so the downstream contract can be tested
before the encoder adapter is available.
"""

from typing import NamedTuple

import torch
from torch import nn


class StateInterpreterOutput(NamedTuple):
    """Interpretable heads and the compact state passed to an RL agent."""

    health_index: torch.Tensor
    stage_logits: torch.Tensor
    stage_probabilities: torch.Tensor
    rl_state: torch.Tensor


class MLPStateInterpreter(nn.Module):
    """Single-window baseline mapping a VibFM embedding to an RL state."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_stages: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_stages < 2:
            raise ValueError("num_stages must be at least 2")

        self.embedding_dim = embedding_dim
        self.num_stages = num_stages
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.health_head = nn.Linear(hidden_dim, 1)
        self.stage_head = nn.Linear(hidden_dim, num_stages)

    @property
    def rl_state_dim(self) -> int:
        """One health indicator plus one probability per degradation stage."""

        return 1 + self.num_stages

    def forward(self, z_health: torch.Tensor) -> StateInterpreterOutput:
        if z_health.ndim != 2:
            raise ValueError(
                "z_health must have shape [batch, embedding_dim], "
                f"got {tuple(z_health.shape)}"
            )
        if z_health.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected embedding_dim={self.embedding_dim}, "
                f"got {z_health.shape[-1]}"
            )

        features = self.backbone(z_health)
        health_index = torch.sigmoid(self.health_head(features))
        stage_logits = self.stage_head(features)
        stage_probabilities = torch.softmax(stage_logits, dim=-1)
        rl_state = torch.cat((health_index, stage_probabilities), dim=-1)
        return StateInterpreterOutput(
            health_index=health_index,
            stage_logits=stage_logits,
            stage_probabilities=stage_probabilities,
            rl_state=rl_state,
        )
