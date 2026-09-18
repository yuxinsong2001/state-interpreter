"""Minimal TS2Vec encoder and hierarchical contrastive objective.

Adapted from the official MIT-licensed implementation at commit
``b0088e14a99706c05451316dc6db8d3da9351163``. This module retains the
encoder and loss semantics needed for the protected XJTU-SY baseline while
leaving experiment policy, data access, and evaluation in local code.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional


class SamePadConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        self.remove = 1 if receptive_field % 2 == 0 else 0
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=receptive_field // 2,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.conv(values)
        return output[:, :, : -self.remove] if self.remove else output


class TS2VecConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        final: bool,
    ) -> None:
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values if self.projector is None else self.projector(values)
        values = self.conv1(functional.gelu(values))
        values = self.conv2(functional.gelu(values))
        return values + residual


class TS2VecDilatedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TS2VecConvBlock(
                    channels[index - 1] if index > 0 else in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=2**index,
                    final=index == len(channels) - 1,
                )
                for index, out_channels in enumerate(channels)
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.blocks(values)


class TS2VecEncoder(nn.Module):
    """Timestamp-level TS2Vec encoder with official masking semantics."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int = 320,
        hidden_dims: int = 64,
        depth: int = 10,
        mask_probability: float = 0.5,
    ) -> None:
        super().__init__()
        if min(input_dims, output_dims, hidden_dims, depth) <= 0:
            raise ValueError("TS2Vec dimensions and depth must be positive")
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must be in [0, 1]")
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.mask_probability = mask_probability
        self.input_projection = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = TS2VecDilatedEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
        )
        self.representation_dropout = nn.Dropout(0.1)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_dims:
            raise ValueError(
                f"expected [batch, timestamps, {self.input_dims}], "
                f"got {tuple(values.shape)}"
            )
        finite_mask = torch.isfinite(values).all(dim=-1)
        clean = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        projected = self.input_projection(clean)
        if mask is None:
            if self.training:
                mask = torch.rand(
                    projected.shape[:2], device=projected.device
                ) < self.mask_probability
            else:
                mask = torch.ones(
                    projected.shape[:2], dtype=torch.bool, device=projected.device
                )
        if mask.shape != projected.shape[:2]:
            raise ValueError("mask must have shape [batch, timestamps]")
        mask = mask & finite_mask
        projected = projected.masked_fill(~mask.unsqueeze(-1), 0.0)
        encoded = self.feature_extractor(projected.transpose(1, 2))
        return self.representation_dropout(encoded).transpose(1, 2)


def instance_contrastive_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    batch, timestamps = first.shape[:2]
    if batch == 1:
        return first.new_tensor(0.0)
    joined = torch.cat((first, second), dim=0).transpose(0, 1)
    similarity = torch.matmul(joined, joined.transpose(1, 2))
    logits = torch.tril(similarity, diagonal=-1)[:, :, :-1]
    logits += torch.triu(similarity, diagonal=1)[:, :, 1:]
    logits = -functional.log_softmax(logits, dim=-1)
    indices = torch.arange(batch, device=first.device)
    return (
        logits[:, indices, batch + indices - 1].mean()
        + logits[:, batch + indices, indices].mean()
    ) / 2.0


def temporal_contrastive_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    timestamps = first.shape[1]
    if timestamps == 1:
        return first.new_tensor(0.0)
    joined = torch.cat((first, second), dim=1)
    similarity = torch.matmul(joined, joined.transpose(1, 2))
    logits = torch.tril(similarity, diagonal=-1)[:, :, :-1]
    logits += torch.triu(similarity, diagonal=1)[:, :, 1:]
    logits = -functional.log_softmax(logits, dim=-1)
    indices = torch.arange(timestamps, device=first.device)
    return (
        logits[:, indices, timestamps + indices - 1].mean()
        + logits[:, timestamps + indices, indices].mean()
    ) / 2.0


def hierarchical_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    alpha: float = 0.5,
    temporal_unit: int = 0,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("contrastive views must share [batch, timestamps, dims]")
    if not 0.0 <= alpha <= 1.0 or temporal_unit < 0:
        raise ValueError("invalid hierarchical contrastive settings")
    loss = first.new_tensor(0.0)
    depth = 0
    while first.shape[1] > 1:
        if alpha:
            loss = loss + alpha * instance_contrastive_loss(first, second)
        if depth >= temporal_unit and alpha < 1.0:
            loss = loss + (1.0 - alpha) * temporal_contrastive_loss(first, second)
        depth += 1
        first = functional.max_pool1d(first.transpose(1, 2), 2).transpose(1, 2)
        second = functional.max_pool1d(second.transpose(1, 2), 2).transpose(1, 2)
    if first.shape[1] == 1 and alpha:
        loss = loss + alpha * instance_contrastive_loss(first, second)
        depth += 1
    return loss / depth


def take_per_row(values: torch.Tensor, offsets: np.ndarray, length: int) -> torch.Tensor:
    columns = torch.as_tensor(
        offsets[:, None] + np.arange(length), dtype=torch.long, device=values.device
    )
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    return values[rows, columns]


@dataclass(frozen=True)
class TS2VecContextViews:
    first: torch.Tensor
    second: torch.Tensor
    overlap_length: int


def sample_context_views(
    values: torch.Tensor,
    *,
    temporal_unit: int,
    random_state: np.random.Generator,
) -> TS2VecContextViews:
    """Create official overlapping context views for one training iteration."""

    timestamps = values.shape[1]
    minimum = 2 ** (temporal_unit + 1)
    if timestamps < minimum:
        raise ValueError("sequence is too short for the selected temporal_unit")
    crop_length = int(random_state.integers(minimum, timestamps + 1))
    crop_left = int(random_state.integers(timestamps - crop_length + 1))
    crop_right = crop_left + crop_length
    extended_left = int(random_state.integers(crop_left + 1))
    extended_right = int(random_state.integers(crop_right, timestamps + 1))
    offsets = random_state.integers(
        -extended_left,
        timestamps - extended_right + 1,
        size=values.shape[0],
    )
    first = take_per_row(
        values, offsets + extended_left, crop_right - extended_left
    )
    second = take_per_row(
        values, offsets + crop_left, extended_right - crop_left
    )
    return TS2VecContextViews(
        first=first,
        second=second,
        overlap_length=crop_length,
    )
