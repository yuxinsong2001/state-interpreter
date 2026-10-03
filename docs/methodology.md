# Methodology and Evidence Boundaries

## Evaluation principles

The project follows four rules:

1. **Simple baselines first.** Added temporal complexity must beat an explicit persistence or distance baseline.
2. **Bearing-level separation.** Measurements from a held-out bearing do not enter training, normalization or checkpoint selection.
3. **Causal state construction.** Interpreter outputs use only present and past embeddings.
4. **Facts, observations and hypotheses remain separate.** Failed transfer is reported rather than hidden.

## Metrics

The primary historical metric is Spearman correlation between Level and normalized lifetime. It evaluates temporal ordering, not numerical calibration.

Supporting metrics include:

- first-to-last Level change;
- Trend direction consistency and noise;
- Movement association with absolute Level change;
- reconstruction MSE within a fixed encoder/normalization setting;
- persistence-relative next-step prediction error;
- stage balanced accuracy in representation audits.

## Data-leakage controls

- normalization is fitted on training bearings only;
- train, validation and test bearing sets are checked for overlap;
- locked configurations are not rewritten with execution results;
- execution state is stored under `records/`;
- protected evaluations require explicit confirmation tokens;
- configuration and artifact hashes are recorded where the protocol requires them;
- a completed blind bearing is not reused for tuning.

## Method comparison

### Relative distance baseline

The mean of the early calibration embeddings defines a bearing-specific reference. Level is a trailing mean of Euclidean distance to the reference, Trend is a least-squares slope of recent Level values, and Movement is the distance between consecutive embeddings.

### HMM

The exploratory HMM uses a three-state left-to-right transition structure and Gaussian emissions. It tests whether an explicit monotonic stage model reduces short reversals. The evaluated version became overconfident and collapsed on one bearing, so it was retained only as a baseline.

### Predictive GRU

The causal GRU learns one-step embedding prediction without lifetime supervision. Multi-seed experiments showed stable training but initialization-sensitive hidden-state geometry. A separate persistence comparison found no positive prediction skill in the fixed comparisons.

### Relative State v2

The later integrated state separates a signed training-only degradation direction, causal Trend and residual-GRU Movement surprise. This remains a relative state; its components are not physical quantities.

## Claim boundaries

- Correlation with normalized lifetime does not prove causation or physical damage.
- One blind bearing cannot establish statistical superiority or broad generalization.
- Multiple seeds on the same folds are not independent machine samples.
- Reconstruction quality does not guarantee a health-aware latent space.
- Development-only Condition 3 results must not be presented as blind evaluation.

