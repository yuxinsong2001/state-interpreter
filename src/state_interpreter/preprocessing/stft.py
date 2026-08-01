"""Deterministic log-STFT preprocessing for XJTU-SY vibration signals."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LogSTFTPreprocessor(nn.Module):
    """Convert raw multi-channel waveforms into compact log-magnitude maps.

    No dataset-level normalization is applied here.  This preserves amplitude
    information and keeps train-only normalization as an explicit later step.
    """

    def __init__(
        self,
        *,
        expected_channels: int = 2,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        output_size: tuple[int, int] = (32, 32),
    ) -> None:
        super().__init__()
        if expected_channels <= 0:
            raise ValueError("expected_channels must be positive")
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if win_length <= 0 or win_length > n_fft:
            raise ValueError("win_length must be positive and no greater than n_fft")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if len(output_size) != 2 or any(size <= 0 for size in output_size):
            raise ValueError("output_size must contain two positive integers")

        self.expected_channels = expected_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.output_size = output_size
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, vibration: torch.Tensor) -> torch.Tensor:
        if vibration.ndim != 2:
            raise ValueError(
                "vibration must have shape [channels, samples], "
                f"got {tuple(vibration.shape)}"
            )
        if vibration.shape[0] != self.expected_channels:
            raise ValueError(
                f"expected {self.expected_channels} channels, "
                f"got {vibration.shape[0]}"
            )
        if vibration.shape[1] < self.n_fft:
            raise ValueError(
                f"vibration must contain at least n_fft={self.n_fft} samples"
            )
        if not torch.is_floating_point(vibration):
            vibration = vibration.float()
        if not bool(torch.isfinite(vibration).all()):
            raise ValueError("vibration contains NaN or infinite values")

        window = self.window.to(device=vibration.device, dtype=vibration.dtype)
        spectrum = torch.stft(
            vibration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        log_magnitude = torch.log1p(spectrum.abs())
        resized = F.adaptive_avg_pool2d(
            log_magnitude.unsqueeze(1), self.output_size
        ).squeeze(1)
        if not bool(torch.isfinite(resized).all()):
            raise ValueError("log-STFT produced NaN or infinite values")
        return resized

