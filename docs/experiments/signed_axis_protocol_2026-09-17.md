# Signed degradation-axis State Interpreter protocol (2026-09-17)

## Question

Does the current Euclidean Level lose useful information by treating forward and backward latent motion identically?

## Fixed method

1. Use only Bearing1_1, Bearing1_2 and Bearing1_3 to fit the first principal axis after each training bearing is centred by its first 15 embeddings.
2. Orient the sign toward the mean late-life displacement of those training bearings.
3. For every new bearing, use only its first 15 embeddings as its local reference.
4. Project each later embedding onto the fixed signed axis and apply the same causal 10-step averaging used by the Euclidean baseline.
5. Fit scalar lifetime and three-stage mappings on the three training bearings only.
6. Evaluate the frozen method on Bearing1_4 and Bearing1_5 from step 25 onward.

## Comparisons and gates

- Compare signed Level against the original Euclidean Level using Spearman rho, backward-step fraction, lifetime calibration and three-stage classification.
- Continue with the signed readout only if both evaluation bearings keep a positive direction and the fixed gates in the JSON configuration pass.
- Do not change parameters after seeing evaluation results.

## Interpretation limits

Normalized lifetime is a chronological proxy, not measured damage. Bearing1_4 and Bearing1_5 have been viewed in earlier experiments, so the result is diagnostic rather than a fresh blind test. A successful result would support a better readout, not prove physical health semantics.

