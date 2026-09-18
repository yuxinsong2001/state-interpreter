"""Materialized STFT datasets and train-only channel normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from state_interpreter.adapters import DatasetAdapter
from state_interpreter.preprocessing import LogSTFTPreprocessor


@dataclass(frozen=True)
class ChannelStandardizer:
    """Per-channel statistics fitted only on the training bearings."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, tensors: torch.Tensor, epsilon: float = 1e-6) -> "ChannelStandardizer":
        if tensors.ndim != 4:
            raise ValueError(
                "tensors must have shape [samples, channels, height, width]"
            )
        if tensors.shape[0] == 0:
            raise ValueError("cannot fit standardizer on an empty tensor")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        mean = tensors.mean(dim=(0, 2, 3), keepdim=True)
        std = tensors.std(dim=(0, 2, 3), correction=0, keepdim=True)
        return cls(mean=mean, std=std.clamp_min(epsilon))

    def transform(self, tensors: torch.Tensor) -> torch.Tensor:
        return (tensors - self.mean.to(tensors)) / self.std.to(tensors)

    def inverse_transform(self, tensors: torch.Tensor) -> torch.Tensor:
        return tensors * self.std.to(tensors) + self.mean.to(tensors)


@dataclass(frozen=True)
class MaterializedSTFTData:
    """In-memory time-frequency tensors with traceable identifiers."""

    tensors: torch.Tensor
    episode_ids: tuple[str, ...]
    step_ids: tuple[int, ...]
    source_paths: tuple[str, ...]

    def as_tensor_dataset(
        self, standardizer: ChannelStandardizer | None = None
    ) -> TensorDataset:
        tensors = (
            self.tensors
            if standardizer is None
            else standardizer.transform(self.tensors)
        )
        return TensorDataset(tensors)


def materialize_stft_data(
    adapter: DatasetAdapter,
    preprocessor: LogSTFTPreprocessor,
) -> MaterializedSTFTData:
    tensors: list[torch.Tensor] = []
    episode_ids: list[str] = []
    step_ids: list[int] = []
    source_paths: list[str] = []
    for measurement in adapter.iter_measurements():
        tensors.append(preprocessor(measurement.vibration))
        episode_ids.append(measurement.episode_id)
        step_ids.append(measurement.step_id)
        source_paths.append(str(measurement.metadata.get("source_path", "")))
    if not tensors:
        raise ValueError("adapter yielded no measurements")
    return MaterializedSTFTData(
        tensors=torch.stack(tensors),
        episode_ids=tuple(episode_ids),
        step_ids=tuple(step_ids),
        source_paths=tuple(source_paths),
    )

