"""Streaming frozen Residual GRU; residuals are not health labels."""
from dataclasses import dataclass
import torch
from state_interpreter.residual_gru import ResidualGRUStateInterpreter

@dataclass(frozen=True)
class OnlineGRUOutput:
    next_prediction: torch.Tensor
    prediction_error_mse: float | None
    persistence_error_mse: float | None
    target_step: int

class OnlineResidualGRU:
    """Consume already-normalized embeddings in ONE checkpoint's coordinates.

    Uses an inference copy of the supplied model; does not mutate training mode.
    Reset between episodes. No calibration, scaling or threshold is learned.
    """
    def __init__(self, model: ResidualGRUStateInterpreter):
        import copy
        if not isinstance(model, ResidualGRUStateInterpreter):
            raise TypeError('ResidualGRUStateInterpreter required')
        self._model=copy.deepcopy(model).eval()
        self.reset()

    def reset(self):
        self._hidden=None;self._previous=None;self._prediction=None;self._step=0

    @torch.inference_mode()
    def update(self,z):
        p=next(self._model.parameters())
        if z.ndim!=1 or z.numel()!=self._model.embedding_dim:
            raise ValueError('one embedding vector required')
        if not torch.is_floating_point(z) or not torch.isfinite(z).all():
            raise ValueError('finite floating embedding required')
        if z.device!=p.device or z.dtype!=p.dtype:
            raise ValueError('embedding must match checkpoint dtype and device')
        current=z.detach().clone()
        error=None if self._prediction is None else float((current-self._prediction).square().mean())
        baseline=None if self._previous is None else float((current-self._previous).square().mean())
        hidden,self._hidden=self._model.gru(current.reshape(1,1,-1),self._hidden)
        prediction=current+self._model.prediction_head(hidden[0,0])
        self._previous=current;self._prediction=prediction.clone()
        result=OnlineGRUOutput(prediction.clone(),error,baseline,self._step)
        self._step+=1
        return result
