# State Interpreter for VibFM

Independent research code for mapping fixed VibFM health embeddings to
interpretable, compact states for PHM and reinforcement learning.

## Current baseline

```text
z_health → MLP → health indicator + degradation-stage probabilities → RL state
```

The repository currently contains only the downstream interface and a shape
test using synthetic embeddings. It does not yet contain a VibFM checkpoint,
real embeddings, model training, or scientific evaluation results.

## Development setup

```powershell
python -m pip install -e .[dev]
python -m pytest -q
```

## Integration boundary

- VibFM is treated as a fixed upstream encoder.
- This package accepts `z_health` with shape `[batch, embedding_dim]`.
- Dataset-, VibFM-, and Gearbox-specific code should be implemented as
  adapters rather than imported into the core model.
