"""Aligned one-step forecast evaluation against persistence."""

import math
import torch


def aligned_squared_errors(sequence: torch.Tensor, forecasts: torch.Tensor):
    """Return model/persistence errors for targets 1..N-1.

    forecasts[t] predicts sequence[t+1]. The last forecast has no observed
    target and is excluded. Persistence copies sequence[t] at the same origin.
    """
    if sequence.ndim != 2 or forecasts.shape != sequence.shape or len(sequence) < 2:
        raise ValueError("matching [time>=2, features] tensors required")
    if not torch.is_floating_point(sequence) or not torch.is_floating_point(forecasts):
        raise ValueError("floating tensors required")
    if not torch.isfinite(sequence).all() or not torch.isfinite(forecasts).all():
        raise ValueError("nonfinite sequence or forecast")
    return (forecasts[:-1] - sequence[1:]).square(), (sequence[:-1] - sequence[1:]).square()


def forecast_metrics(gru_errors: torch.Tensor, persistence_errors: torch.Tensor, *, target_start_step: int):
    if gru_errors.ndim != 2 or persistence_errors.shape != gru_errors.shape:
        raise ValueError("matching [targets, features] errors required")
    if not 1 <= target_start_step <= len(gru_errors):
        raise ValueError("target_start_step outside observed target range")
    # Row zero refers to target step 1, not the forecast origin step 0.
    g = gru_errors[target_start_step - 1:].double()
    p = persistence_errors[target_start_step - 1:].double()
    if not torch.isfinite(g).all() or not torch.isfinite(p).all() or (g < 0).any() or (p < 0).any():
        raise ValueError("finite nonnegative squared errors required")
    gm, pm = float(g.mean()), float(p.mean())
    return {
        "target_count": len(g), "gru_mse": gm, "persistence_mse": pm,
        "mse_ratio": gm / pm if pm > 0 else None,
        "skill": 1 - gm / pm if pm > 0 else None,
        "gru_rmse": math.sqrt(gm), "persistence_rmse": math.sqrt(pm),
        "gru_win_fraction": float((g.mean(1) < p.mean(1)).double().mean()),
        "gru_median_step_mse": float(g.mean(1).quantile(0.5)),
        "persistence_median_step_mse": float(p.mean(1).quantile(0.5)),
    }
