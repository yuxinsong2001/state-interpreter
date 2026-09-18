"""Causal stability transforms for early-calibrated engineering features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from state_interpreter.data.feature_sequences import CausalFeatureStandardizer


ScalingMode = Literal["baseline_zscore", "signed_log1p", "clip_20"]


@dataclass(frozen=True)
class StableCausalFeatureScaler:
    """Fit on a fixed early prefix, then apply one preregistered stabilization."""

    standardizer: CausalFeatureStandardizer
    mode: ScalingMode

    @classmethod
    def fit(
        cls,
        sequence: torch.Tensor,
        *,
        calibration_steps: int = 15,
        mode: ScalingMode = "signed_log1p",
    ) -> "StableCausalFeatureScaler":
        if mode not in ("baseline_zscore", "signed_log1p", "clip_20"):
            raise ValueError(f"unsupported scaling mode: {mode}")
        return cls(
            standardizer=CausalFeatureStandardizer.fit(
                sequence, calibration_steps=calibration_steps
            ),
            mode=mode,
        )

    def transform(self, sequence: torch.Tensor) -> torch.Tensor:
        values = self.standardizer.transform(sequence)
        if self.mode == "baseline_zscore":
            output = values
        elif self.mode == "signed_log1p":
            output = torch.sign(values) * torch.log1p(torch.abs(values))
        else:
            output = torch.clamp(values, min=-20.0, max=20.0)
        if not torch.isfinite(output).all():
            raise RuntimeError("stable feature scaling produced non-finite values")
        return output
