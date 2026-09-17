# 2026-09-17 Early-window scale calibration

This immutable experiment tests whether the first 15 embeddings of a new bearing can calibrate signed-axis Level to a common absolute scale.

## Selection result

Training-only leave-one-bearing-out selection chose `none`.

| Candidate | Mean stage balanced accuracy | Mean lifetime MAE | Mean R2 |
|---|---:|---:|---:|
| none | 0.628 | 0.172 | 0.276 |
| early_axis_std | 0.447 | 0.333 | -1.883 |
| early_latent_rms | 0.447 | 0.330 | -1.811 |

The failed scale candidates were not evaluated on Bearing1_4 or Bearing1_5. This preserves the predeclared separation between method selection and evaluation.

## Files

- `experiment_report.json`: hashes, selected method, decision and 11/11 fallacy scan.
- `training_lobo_folds.csv`: all training-only selection folds.
- `training_lobo_summary.csv`: candidate-level selection metrics.
- `evaluation_metrics.csv`: frozen selected-method evaluation.
- `early_scales.csv`: raw and floored calibration scales.
- `evaluation_per_step.csv`: auditable Level trajectories.
- `early_scale_comparison.png`: selected-method evaluation trajectories.

Normalized lifetime is a chronological proxy, not measured damage.

