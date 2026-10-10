"""Two-task PCGrad adaptation for shared GRU parameters.

Based on Yu et al., NeurIPS 2020, Algorithm 1. This is a project
PyTorch adaptation, not the authors' TensorFlow implementation.
No magnitude balancing; independent heads retain their own gradients.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PCGradDiagnostics:
    prediction_norm: float
    level_norm: float
    cosine: float | None
    projected: bool
    zero_norm_skipped: bool
    combined_norm: float


def project_two_gradients(a, b, *, epsilon=1e-12):
    """Project flat vectors using originals; epsilon is a norm cutoff."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive finite epsilon required')
    if a.ndim != 1 or a.shape != b.shape or a.numel() == 0:
        raise ValueError('matching nonempty flat vectors required')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('nonfinite gradients')
    aa, bb, dot = a.square().sum(), b.square().sum(), (a*b).sum()
    if not all(torch.isfinite(x) for x in (aa, bb, dot)):
        raise ValueError('nonfinite gradient reductions')
    an, bn = aa.sqrt(), bb.sqrt()
    skip = float(an) <= epsilon or float(bn) <= epsilon
    cosine = None if skip else float(dot / (an*bn))
    projected = not skip and float(dot) < 0
    ap = a - dot/bb*b if projected else a
    bp = b - dot/aa*a if projected else b
    combined = ap + bp
    if not torch.isfinite(combined).all():
        raise ValueError('nonfinite projected gradients')
    return ap, bp, PCGradDiagnostics(
        float(an), float(bn), cosine, projected, skip,
        float(torch.linalg.vector_norm(combined)),
    )


def pcgrad_shared_gradients(model, prediction_loss, level_loss, *, epsilon=1e-12):
    """Populate grads only. Caller clips afterward, then calls Adam.step()."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive finite epsilon required')
    shared = list(model.core.gru.parameters())
    ph = list(model.core.prediction_head.parameters())
    lh = list(model.level_head.parameters())
    a = torch.autograd.grad(prediction_loss, shared+ph, retain_graph=True)
    b = torch.autograd.grad(level_loss, shared+lh)
    if not all(torch.isfinite(g).all() for g in (*a, *b)):
        raise ValueError('nonfinite gradients')
    n = len(shared)
    af = torch.cat([g.reshape(-1) for g in a[:n]])
    bf = torch.cat([g.reshape(-1) for g in b[:n]])
    ap, bp, diagnostics = project_two_gradients(af, bf, epsilon=epsilon)
    combined = ap+bp
    offset = 0
    for parameter in shared:
        count = parameter.numel()
        parameter.grad = combined[offset:offset+count].reshape_as(parameter).detach().clone()
        offset += count
    for parameter, gradient in zip(ph, a[n:]):
        parameter.grad = gradient.detach().clone()
    for parameter, gradient in zip(lh, b[n:]):
        parameter.grad = gradient.detach().clone()
    return diagnostics
