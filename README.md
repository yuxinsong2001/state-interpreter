# State Interpreter

Independent research code for learning compact vibration representations and
mapping them to interpretable states for PHM and reinforcement learning.

## Current research path

```text
XJTU-SY vibration
→ small convolutional autoencoder
→ z (8 or 16 dimensions)
→ latent-space State Interpreter
→ later migration to VibFM
```

The autoencoder is a replaceable upstream module.  The interpreter consumes a
standard embedding tensor and must not depend on autoencoder internals.  This
keeps the later replacement `AutoEncoder → VibFM` local to the encoder adapter.

## Current baseline

```text
z_health → MLP → health indicator + degradation-stage probabilities → RL state
```

The repository contains shape-tested model interfaces only. It does not yet
contain extracted XJTU-SY data, model training, real embeddings, or scientific
evaluation results.

## Development setup

```powershell
python -m pip install -e .[dev]
python -m pytest -q
```

## Integration boundary

- `SmallConvAutoEncoder` accepts `[batch, 2, 32, 32]` and exposes `encode(x)`.
- The State Interpreter accepts embeddings with shape `[batch, embedding_dim]`.
- Dataset-, VibFM-, and Gearbox-specific code should be implemented as
  adapters rather than imported into the core model.
