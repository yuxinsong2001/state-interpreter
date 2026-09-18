"""Causal health encoder with source-domain adversarial regularization.

The gradient-reversal design follows the DANN principle introduced by Ganin
et al. (2016).  This module is a small project-specific implementation rather
than a verbatim copy of an upstream library.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


class GradientReversal(nn.Module):
    """Identity in forward propagation and sign reversal in backpropagation."""

    def forward(self, inputs: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
        if coefficient < 0.0:
            raise ValueError("gradient-reversal coefficient must be non-negative")
        return _GradientReversalFunction.apply(inputs, coefficient)


class DomainAdversarialOutput(NamedTuple):
    health: torch.Tensor
    embedding: torch.Tensor
    domain_logits: torch.Tensor


class DomainAdversarialHealthEncoder(nn.Module):
    """Unidirectional GRU with health and source-bearing identity heads."""

    def __init__(
        self,
        *,
        n_features: int = 65,
        hidden_size: int = 32,
        embedding_size: int = 16,
        domain_hidden_size: int = 32,
        domain_count: int = 2,
    ) -> None:
        super().__init__()
        if min(
            n_features,
            hidden_size,
            embedding_size,
            domain_hidden_size,
            domain_count,
        ) <= 0:
            raise ValueError("all model dimensions must be positive")
        if domain_count < 2:
            raise ValueError("at least two source domains are required")
        self.n_features = n_features
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
        )
        self.health_head = nn.Linear(embedding_size, 1)
        self.gradient_reversal = GradientReversal()
        self.domain_head = nn.Sequential(
            nn.Linear(embedding_size, domain_hidden_size),
            nn.ReLU(),
            nn.Linear(domain_hidden_size, domain_count),
        )

    def forward(
        self,
        windows: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        grl_coefficient: float = 1.0,
    ) -> DomainAdversarialOutput:
        if windows.ndim != 3 or windows.shape[-1] != self.n_features:
            raise ValueError(
                f"expected [batch, steps, {self.n_features}], got {tuple(windows.shape)}"
            )
        if windows.shape[1] < 1 or not torch.isfinite(windows).all():
            raise ValueError("windows must be non-empty and finite")
        if lengths is None:
            _, hidden = self.gru(windows)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device="cpu")
            if lengths.shape != (windows.shape[0],):
                raise ValueError("lengths must contain one value per window")
            if torch.any(lengths < 1) or torch.any(lengths > windows.shape[1]):
                raise ValueError("lengths must fit the window size")
            packed = pack_padded_sequence(
                windows,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
        embedding = self.projector(hidden[-1])
        health = self.health_head(embedding)
        reversed_embedding = self.gradient_reversal(embedding, grl_coefficient)
        domain_logits = self.domain_head(reversed_embedding)
        return DomainAdversarialOutput(health, embedding, domain_logits)


def dann_coefficient(
    progress: float,
    *,
    maximum: float = 0.1,
    steepness: float = 10.0,
) -> float:
    """Logistic warm-start schedule used for the adversarial coefficient."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    if maximum < 0.0 or steepness <= 0.0:
        raise ValueError("invalid schedule parameters")
    value = 2.0 / (1.0 + torch.exp(torch.tensor(-steepness * progress))) - 1.0
    return float(maximum * value)

