"""Causal streaming wrapper for the fixed dual-head relative GRU."""
import copy
from dataclasses import dataclass
from collections import deque
import torch
from .gru_state_readout import GRUStateReadoutInterpreter


@dataclass(frozen=True)
class OnlineStateOutput:
    step: int
    phase: str
    level: float | None
    trend: float | None
    movement: float | None


class OnlineGRUStateInterpreter:
    """Raw encoder z -> episode reference -> train scaling -> frozen GRU.

    Calibration buffers first 15 points, then replays them using the newly
    available reference. No state is published before step 25. Movement is
    the RMSE of the prior forecast in standardized embedding coordinates.
    No health threshold, damage estimate or maintenance action is provided.
    """
    def __init__(self, model, train_mean, train_std, level_target_scale):
        if not isinstance(model, GRUStateReadoutInterpreter):
            raise TypeError('dual-head GRU required')
        self._model = copy.deepcopy(model).eval()
        p = next(self._model.parameters())
        self._mean = torch.as_tensor(train_mean,device=p.device,dtype=p.dtype).clone()
        self._std = torch.as_tensor(train_std,device=p.device,dtype=p.dtype).clone()
        self._scale = torch.as_tensor(level_target_scale,device=p.device,dtype=p.dtype).clone()
        d = model.embedding_dim
        if self._mean.shape != (d,) or self._std.shape != (d,):
            raise ValueError('matching train statistics required')
        if not all(torch.isfinite(x).all() for x in (self._mean,self._std,self._scale)):
            raise ValueError('finite train statistics required')
        if not (self._std>0).all() or self._scale.numel()!=1 or self._scale.item()<=0:
            raise ValueError('positive std and scalar target scale required')
        self.reset()

    def reset(self):
        self._step=0
        self._buffer=[]
        self._center=None
        self._hidden=None
        self._prediction=None
        self._levels=deque(maxlen=10)

    def _consume(self, raw, step):
        z=(raw-self._center-self._mean)/self._std
        movement=None if self._prediction is None else float((z-self._prediction).square().mean().sqrt())
        hidden,self._hidden=self._model.core.gru(z.reshape(1,1,-1),self._hidden)
        self._prediction=(z+self._model.core.prediction_head(hidden[0,0])).clone()
        level=float(self._model.level_head(hidden[0,0]).reshape(())*self._scale.reshape(()))
        if step>=15:self._levels.append(level)
        return level,movement

    @torch.inference_mode()
    def update(self, z):
        p=next(self._model.parameters())
        if not isinstance(z,torch.Tensor) or z.shape!=(self._model.embedding_dim,):
            raise ValueError('one embedding tensor required')
        if z.dtype!=p.dtype or z.device!=p.device or not torch.isfinite(z).all():
            raise ValueError('finite embedding matching model dtype/device required')
        raw=z.detach().clone()
        step=self._step
        if step<15:
            self._buffer.append(raw)
            if step==14:
                self._center=torch.stack(self._buffer).mean(0)
                for i,value in enumerate(self._buffer):self._consume(value,i)
                self._buffer=[]
            result=OnlineStateOutput(step,'CALIBRATING' if step<14 else 'WARMING',None,None,None)
        else:
            level,movement=self._consume(raw,step)
            if step<25:
                result=OnlineStateOutput(step,'WARMING',None,None,None)
            else:
                values=torch.tensor(list(self._levels),dtype=p.dtype,device=p.device)
                x=torch.arange(len(values),dtype=p.dtype,device=p.device)
                x-=x.mean()
                trend=float((x*(values-values.mean())).sum()/x.square().sum())
                result=OnlineStateOutput(step,'READY',level,trend,movement)
        self._step+=1
        return result
