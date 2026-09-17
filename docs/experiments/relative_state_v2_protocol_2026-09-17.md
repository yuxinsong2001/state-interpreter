# Relative State Interpreter v2 integration protocol (2026-09-17)

## State definition

- **Level**: signed projection onto the frozen degradation axis, relative to the current bearing's first 15 embeddings and causally averaged over at most 10 outputs.
- **Trend**: least-squares slope of the last at most 10 Level values.
- **Movement**: ensemble Residual-GRU one-step prediction surprise in standardized latent space, divided by the median training-bearing surprise from target step 15 onward.

## Frozen inputs

The signed axis and three Residual-GRU checkpoints from the completed experiments are loaded without retraining. Checkpoint normalization tensors must be identical. The movement reference is fitted only on Bearing1_1–Bearing1_3.

## Required integrity checks

1. Online Level exactly reproduces the archived signed-axis Level.
2. Prefix execution produces identical states to full chronological execution.
3. Every READY state is finite and Movement is non-negative.
4. Reset prevents state transfer between bearings.
5. The training Movement median equals one by construction.
6. Frozen artifact hashes remain unchanged.

No monotonicity requirement is imposed on Trend or Movement. They describe local direction and prediction surprise, not damage severity.

