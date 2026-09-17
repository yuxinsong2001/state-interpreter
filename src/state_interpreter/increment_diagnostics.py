"""Diagnostics for one-step latent increments."""

from __future__ import annotations

import math

import torch


def aligned_increments(
    sequence: torch.Tensor,
    forecasts: torch.Tensor,
    *,
    target_start_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return observed and predicted increments for the selected targets.

    ``forecasts[t]`` predicts ``sequence[t + 1]``.  Therefore both increments
    use ``sequence[t]`` as their origin.  Target step numbering starts at one.
    """
    if sequence.ndim != 2 or forecasts.shape != sequence.shape or len(sequence) < 2:
        raise ValueError("matching [time>=2, features] tensors required")
    if not torch.is_floating_point(sequence) or not torch.is_floating_point(forecasts):
        raise ValueError("floating tensors required")
    if not torch.isfinite(sequence).all() or not torch.isfinite(forecasts).all():
        raise ValueError("nonfinite sequence or forecast")
    if not 1 <= target_start_step < len(sequence):
        raise ValueError("target_start_step outside observed target range")
    origins = sequence[target_start_step - 1 : -1]
    true_increment = sequence[target_start_step:] - origins
    predicted_increment = forecasts[target_start_step - 1 : -1] - origins
    return true_increment, predicted_increment


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def increment_metrics(true_increment: torch.Tensor, predicted_increment: torch.Tensor) -> dict:
    """Summarise bias, direction and magnitude of predicted increments.

    The oracle correction subtracts the mean error computed on the complete
    evaluated sequence.  It is a post-hoc diagnostic and must not be used as a
    deployable result.
    """
    if true_increment.ndim != 2 or predicted_increment.shape != true_increment.shape:
        raise ValueError("matching [targets, features] increments required")
    if len(true_increment) < 1:
        raise ValueError("at least one target required")
    if not torch.isfinite(true_increment).all() or not torch.isfinite(predicted_increment).all():
        raise ValueError("finite increments required")

    true = true_increment.double()
    predicted = predicted_increment.double()
    error = predicted - true
    bias = error.mean(0)
    corrected_error = error - bias

    persistence_mse = float(true.square().mean())
    raw_mse = float(error.square().mean())
    bias_mse = float(bias.square().mean())
    corrected_mse = float(corrected_error.square().mean())
    true_rms = math.sqrt(persistence_mse)
    predicted_rms = math.sqrt(float(predicted.square().mean()))

    flat_dot = float((true * predicted).sum())
    true_energy = float(true.square().sum())
    predicted_energy = float(predicted.square().sum())
    cosine = flat_dot / math.sqrt(true_energy * predicted_energy) if true_energy > 0 and predicted_energy > 0 else None
    gain = flat_dot / true_energy if true_energy > 0 else None

    step_dot = (true * predicted).sum(1)
    step_true_norm = true.norm(dim=1)
    step_predicted_norm = predicted.norm(dim=1)
    valid = (step_true_norm > 0) & (step_predicted_norm > 0)
    step_cosines = step_dot[valid] / (step_true_norm[valid] * step_predicted_norm[valid])

    return {
        "target_count": len(true),
        "feature_count": true.shape[1],
        "persistence_mse": persistence_mse,
        "raw_mse": raw_mse,
        "raw_mse_ratio": _safe_ratio(raw_mse, persistence_mse),
        "raw_skill": 1 - raw_mse / persistence_mse if persistence_mse > 0 else None,
        "bias_mse": bias_mse,
        "bias_fraction_of_raw_mse": _safe_ratio(bias_mse, raw_mse),
        "oracle_bias_corrected_mse": corrected_mse,
        "oracle_bias_corrected_mse_ratio": _safe_ratio(corrected_mse, persistence_mse),
        "oracle_bias_corrected_skill": 1 - corrected_mse / persistence_mse if persistence_mse > 0 else None,
        "true_increment_rms": true_rms,
        "predicted_increment_rms": predicted_rms,
        "magnitude_ratio": _safe_ratio(predicted_rms, true_rms),
        "flattened_cosine": cosine,
        "through_origin_gain": gain,
        "positive_dot_fraction": float((step_dot > 0).double().mean()),
        "valid_step_cosine_count": int(valid.sum()),
        "mean_step_cosine": float(step_cosines.mean()) if len(step_cosines) else None,
        "median_step_cosine": float(step_cosines.quantile(0.5)) if len(step_cosines) else None,
        "mean_error_bias": [float(value) for value in bias],
    }
