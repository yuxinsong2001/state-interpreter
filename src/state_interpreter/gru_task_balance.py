"""Train-only shared-GRU gradient magnitude balancing.

This is an explicit engineering rule, not conflict projection or PCGrad.
It does not guarantee simultaneous improvement of both objectives.
"""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class TaskBalanceDiagnostics:
    prediction_norm: float
    level_norm: float
    prediction_weight: float
    level_weight: float
    cosine: float | None
    fallback: bool


def balanced_shared_gradients(model, prediction_loss, level_loss, *, epsilon=1e-12):
    """Populate grads; caller subsequently clips then calls optimizer.step().

    Shared g = mean(norms)*(g_pred/n_pred + g_level/n_level).
    If norms equal, this is exactly the ordinary sum. Heads keep their own
    unweighted loss gradients. Zero/nonfinite handling never updates weights.
    """
    if not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('positive finite epsilon required')
    shared=list(model.core.gru.parameters())
    pred_head=list(model.core.prediction_head.parameters())
    level_head=list(model.level_head.parameters())
    a=torch.autograd.grad(prediction_loss,shared+pred_head,retain_graph=True)
    b=torch.autograd.grad(level_loss,shared+level_head)
    ns=len(shared)
    if not all(torch.isfinite(x).all() for x in [*a,*b]):
        raise ValueError('nonfinite gradients')
    an=torch.stack([g.square().sum() for g in a[:ns]]).sum().sqrt()
    bn=torch.stack([g.square().sum() for g in b[:ns]]).sum().sqrt()
    n1,n2=float(an),float(bn)
    fallback=n1<=epsilon or n2<=epsilon
    cosine=None if fallback else float(sum((x*y).sum() for x,y in zip(a[:ns],b[:ns]))/(an*bn))
    if fallback:w1=w2=1.
    else:
        magnitude=(n1+n2)/2
        w1,w2=magnitude/n1,magnitude/n2
    for p,x,y in zip(shared,a[:ns],b[:ns]):p.grad=(w1*x+w2*y).detach().clone()
    for p,g in zip(pred_head,a[ns:]):p.grad=g.detach().clone()
    for p,g in zip(level_head,b[ns:]):p.grad=g.detach().clone()
    return TaskBalanceDiagnostics(n1,n2,w1,w2,cosine,fallback)
