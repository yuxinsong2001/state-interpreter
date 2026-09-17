# Early-window scale calibration protocol (2026-09-17)

## Question

Can the first 15 embeddings of a new bearing provide enough information to place its signed degradation-axis Level on a common cross-bearing scale?

## Candidate methods fixed before evaluation

- `none`: signed-axis Level without per-bearing scaling.
- `early_axis_std`: divide by the sample standard deviation of early projections on the fixed axis.
- `early_latent_rms`: divide by the RMS norm of the early centred latent cloud.

A lower bound equal to 10% of the median scale of the current training folds prevents division by a near-zero scale. This floor is never fitted on the held bearing.

## Selection and evaluation

1. Within Bearing1_1–Bearing1_3, run leave-one-bearing-out folds. Refit the signed axis, scale floor, scalar lifetime Ridge and stage classifier in each fold.
2. Select the method by highest mean stage balanced accuracy, then lowest mean lifetime MAE, then the predeclared candidate order.
3. Refit the selected method on all three training bearings.
4. Evaluate once on Bearing1_4 and Bearing1_5; their later samples are not used for calibration.

The primary question is absolute transfer (stage accuracy and MAE), not Spearman ordering. Results are diagnostic because only three training and two previously viewed evaluation bearings are available.

