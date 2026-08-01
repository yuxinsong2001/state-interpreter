"""Small convolutional autoencoder used to study low-dimensional states.

This module is deliberately independent from the State Interpreter.  Its only
contract is to convert a two-channel 32 x 32 time-frequency representation into
a fixed-size latent vector and to reconstruct the original input during
unsupervised training.
"""

from typing import NamedTuple

import torch
from torch import nn


class AutoEncoderOutput(NamedTuple):
    """Reconstruction and latent representation returned during training."""

    reconstruction: torch.Tensor
    z: torch.Tensor


class SmallConvAutoEncoder(nn.Module):
    """Compact 32 x 32 convolutional autoencoder with an 8/16-D latent space."""

    def __init__(self, input_channels: int = 2, latent_dim: int = 8) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.input_channels = input_channels
        self.latent_dim = latent_dim

        self.encoder_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.to_latent = nn.Linear(64 * 4 * 4, latent_dim)

        self.from_latent = nn.Linear(latent_dim, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, input_channels, kernel_size=4, stride=2, padding=1
            ),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "x must have shape [batch, channels, 32, 32], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_channels or x.shape[2:] != (32, 32):
            raise ValueError(
                f"expected [batch, {self.input_channels}, 32, 32], "
                f"got {tuple(x.shape)}"
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return one latent vector per input measurement."""

        self._validate_input(x)
        features = self.encoder_backbone(x)
        return self.to_latent(features.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of time-frequency inputs from latent vectors."""

        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"expected z with shape [batch, {self.latent_dim}], "
                f"got {tuple(z.shape)}"
            )
        features = self.from_latent(z).reshape(-1, 64, 4, 4)
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> AutoEncoderOutput:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return AutoEncoderOutput(reconstruction=reconstruction, z=z)

