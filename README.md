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
ordered z
→ episode-specific early calibration
→ relative distance + causal temporal features
→ [level, trend, movement]
```

`RelativeTemporalStateInterpreter` is the current unsupervised online baseline.
It has an explicit `CALIBRATING → READY` lifecycle, emits no formal state during
calibration, and requires `reset()` before a new episode. The existing
`MLPStateInterpreter` remains a shape-tested option for future supervised or
multi-task experiments; it is not the selected first baseline.

The first XJTU-SY engineering run now includes a trained `z=8` AutoEncoder,
ordered latent extraction, health-indicator comparison, and online replay on
five bearings. These results establish engineering feasibility and temporal
consistency only; they do not prove that the output is a physical health state.

```python
from state_interpreter import RelativeTemporalStateInterpreter

interpreter = RelativeTemporalStateInterpreter(
    embedding_dim=8,
    calibration_steps=10,
    temporal_window=5,
)

for z_t in ordered_embeddings:
    output = interpreter.update(z_t)
    if output is not None:
        state_t = output.state  # [level, trend, movement]
```

## Development setup

```powershell
python -m pip install -e .[dev]
python -m pytest -q
```

## XJTU-SY raw-data smoke test

The dataset adapter reads raw CSV files in numeric measurement order and emits
two-channel tensors without applying STFT or normalization:

```powershell
python scripts/smoke_test_xjtu_sy.py `
  --root "D:\path\to\XJTU-SY_Bearing_Datasets" `
  --condition 35Hz12kN `
  --bearing Bearing1_1 `
  --limit 2
```

## Log-STFT inspection

The first preprocessing baseline uses a Hann window, `n_fft=1024`,
`win_length=1024`, `hop_length=512`, `log1p` magnitude compression, and adaptive
average pooling to `[2, 32, 32]`. No dataset-level normalization is applied at
this stage.

```powershell
python scripts/inspect_xjtu_stft.py `
  --root "D:\path\to\XJTU-SY_Bearing_Datasets" `
  --condition 35Hz12kN `
  --bearing Bearing1_1 `
  --output outputs/xjtu_stft_bearing1_1.png
```

## First AutoEncoder baseline

The first engineering baseline uses one operating condition and a bearing-wise
split: Bearings 1_1–1_3 for training, 1_4 for validation, and 1_5 as an
untouched holdout. Channel normalization is fitted on the training bearings
only.

```powershell
python scripts/train_autoencoder_baseline.py `
  --root "D:\path\to\XJTU-SY_Bearing_Datasets" `
  --output-dir "D:\path\to\run-output" `
  --condition 35Hz12kN `
  --latent-dim 8 `
  --epochs 5
```

## Integration boundary

- `SmallConvAutoEncoder` accepts `[batch, 2, 32, 32]` and exposes `encode(x)`.
- The State Interpreter accepts embeddings with shape `[batch, embedding_dim]`.
- Dataset-, VibFM-, and Gearbox-specific code should be implemented as
  adapters rather than imported into the core model.
