"""Abstract integration boundaries.

Concrete implementations belong to the corresponding external-system
integration and must convert their output into the contracts defined here.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch

from state_interpreter.contracts import (
    EmbeddingSample,
    RawMeasurement,
    StateTransition,
)


class VibFMAdapter(ABC):
    """Convert vibration tensors into fixed VibFM health embeddings."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of one returned ``z_health`` vector."""

    @abstractmethod
    def encode(self, vibration: torch.Tensor) -> torch.Tensor:
        """Return one ``z_health`` vector with shape ``[embedding_dim]``."""

    def transform(self, measurement: RawMeasurement) -> EmbeddingSample:
        """Encode a standard measurement while preserving its metadata."""

        with torch.no_grad():
            z_health = self.encode(measurement.vibration)
        if z_health.shape != (self.embedding_dim,):
            raise ValueError(
                f"adapter declared embedding_dim={self.embedding_dim}, "
                f"but returned shape {tuple(z_health.shape)}"
            )
        return EmbeddingSample(measurement=measurement, z_health=z_health)


class DatasetAdapter(ABC):
    """Read an ordered measured dataset without applying VibFM."""

    @abstractmethod
    def iter_measurements(self) -> Iterable[RawMeasurement]:
        """Yield measurements ordered within each episode/run."""


class GearboxAdapter(ABC):
    """Collect standardized transitions from a Gearbox/RL environment."""

    @abstractmethod
    def collect_episode(
        self,
        encoder: VibFMAdapter,
        *,
        max_steps: int | None = None,
    ) -> Iterable[StateTransition]:
        """Yield one complete, ordered episode of encoded transitions."""
