"""Train-source-only sampling and MMD diagnostics; no model training."""

from __future__ import annotations

import numpy as np
import torch

from state_interpreter.representation_diagnostics import evenly_spaced_indices


def paired_epoch_indices(
    lengths: tuple[int, int], *, seed: int, bins: int = 5, per_cell: int = 6,
    batches: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return endpoint, source and time-bin arrays shaped [batches, batch]."""
    if len(lengths) != 2 or min(lengths) < 300 or bins != 5 or per_cell != 6 or batches != 10:
        raise ValueError("preflight requires two sources and the fixed 5x6x10 design")
    rng = np.random.default_rng(seed)
    cells: dict[tuple[int, int], np.ndarray] = {}
    for source, length in enumerate(lengths):
        endpoints = evenly_spaced_indices(length, 300)
        labels = np.minimum(bins - 1, (endpoints / (length - 1) * bins).astype(int))
        for time_bin in range(bins):
            available = rng.permutation(endpoints[labels == time_bin])
            if available.size < per_cell:
                raise ValueError("time bin has too few distinct endpoints")
            cells[source, time_bin] = np.resize(available, batches * per_cell)
    endpoint_batches = []
    source_batches = []
    bin_batches = []
    for batch in range(batches):
        endpoints, sources, labels = [], [], []
        for source in range(2):
            for time_bin in range(bins):
                selected = cells[source, time_bin][batch * per_cell:(batch + 1) * per_cell]
                endpoints.extend(selected.tolist())
                sources.extend([source] * per_cell)
                labels.extend([time_bin] * per_cell)
        endpoint_batches.append(endpoints)
        source_batches.append(sources)
        bin_batches.append(labels)
    return (np.asarray(endpoint_batches, dtype=np.int64),
            np.asarray(source_batches, dtype=np.int64),
            np.asarray(bin_batches, dtype=np.int64))


def median_squared_distance(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.ndim != 2 or embedding.shape[0] < 2 or not torch.isfinite(embedding).all():
        raise ValueError("embedding must be finite [samples, dimensions]")
    with torch.no_grad():
        distance = torch.pdist(embedding.detach(), p=2).square()
        positive = distance[distance > 0]
        if positive.numel() == 0:
            raise ValueError("degenerate embedding distance")
        scale = positive.median()
        if not torch.isfinite(scale) or scale <= 1e-12:
            raise ValueError("degenerate kernel scale")
        return scale


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, squared_scale: torch.Tensor) -> torch.Tensor:
    """Biased squared MMD; differentiable in x/y, fixed scale."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(x.shape[0], y.shape[0]) < 2:
        raise ValueError("MMD needs two nontrivial equally dimensional groups")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("MMD inputs must be finite")
    if not torch.isfinite(squared_scale) or squared_scale <= 0:
        raise ValueError("kernel scale must be positive and finite")
    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(a, b).square() / (2 * squared_scale))
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()


def paired_mmd2(
    embedding: torch.Tensor, sources: torch.Tensor, bins: torch.Tensor,
    squared_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embedding.shape[0] != sources.numel() or embedding.shape[0] != bins.numel():
        raise ValueError("MMD labels and embedding rows differ")
    if set(sources.tolist()) != {0, 1} or set(bins.tolist()) != set(range(5)):
        raise ValueError("MMD requires two sources and five bins")
    global_loss = rbf_mmd2(embedding[sources == 0], embedding[sources == 1], squared_scale)
    local = []
    for time_bin in range(5):
        local.append(rbf_mmd2(
            embedding[(sources == 0) & (bins == time_bin)],
            embedding[(sources == 1) & (bins == time_bin)], squared_scale,
        ))
    return global_loss, torch.stack(local).mean()
