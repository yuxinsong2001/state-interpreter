"""Validated CPU loading for the fixed relative-state GRU prototype."""
from pathlib import Path
import torch
from .gru_state_readout import GRUStateReadoutInterpreter
from .online_gru_state import OnlineGRUStateInterpreter


def load_online_gru_checkpoint(path):
    """Return a fresh reset interpreter; never fit statistics on new input.

    The caller must supply embeddings in the encoder coordinates used by
    training. The checkpoint does not establish physical health semantics.
    Only the current one-layer 15/10/25 prototype contract is supported.
    """
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    required={'model_state_dict','config','train_mean','train_std','level_target_scale'}
    if not isinstance(checkpoint,dict) or not required.issubset(checkpoint):
        raise ValueError('complete state checkpoint and training statistics required')
    cfg=checkpoint['config']
    if not isinstance(cfg,dict) or any(cfg.get(k)!=v for k,v in [('calibration_steps',15),('temporal_window',10),('score_start',25)]):
        raise ValueError('checkpoint must match fixed 15/10/25 contract')
    if cfg.get('level_only'):
        raise ValueError('Level-only diagnostic checkpoint is not a full state model')
    state=checkpoint['model_state_dict']
    weight=state.get('core.gru.weight_ih_l0')
    if not isinstance(weight,torch.Tensor) or weight.ndim!=2 or weight.shape[0]%3:
        raise ValueError('valid GRU state required')
    model=GRUStateReadoutInterpreter(embedding_dim=weight.shape[1],hidden_dim=weight.shape[0]//3)
    if not all(isinstance(value,torch.Tensor) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError('finite model tensors required')
    model.load_state_dict(state,strict=True)
    return OnlineGRUStateInterpreter(model,checkpoint['train_mean'],checkpoint['train_std'],checkpoint['level_target_scale'])
