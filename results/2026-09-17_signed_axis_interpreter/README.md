# 2026-09-17 Signed-axis State Interpreter

This immutable result directory contains the fixed-protocol comparison between the original bearing-relative Euclidean Level and a training-only signed degradation-axis Level.

## Main result

- Bearing1_4: Spearman rho 0.8531 → 0.9269; backward-step fraction 40.63% → 36.46%.
- Bearing1_5: Spearman rho 0.9713 → 0.9896; backward-step fraction 15.38% → 11.54%.
- All predeclared continuation gates passed.
- Stage balanced accuracy and absolute lifetime calibration did not improve meaningfully; this method improves relative ordering, not physical state semantics.

## Files

- `experiment_report.json`: hashes, gate decisions and 11/11 fallacy scan.
- `axis.json`: frozen unit degradation direction.
- `trajectory_metrics.csv`: per-bearing ranking and trajectory metrics.
- `stage_metrics.csv`: fixed train-to-evaluation three-stage results.
- `lifetime_calibration_metrics.csv`: fixed scalar lifetime mapping results.
- `per_step_levels.csv`: auditable Euclidean and signed Level values.
- `signed_axis_comparison.png`: evaluation-bearing trajectories and signed step changes.

The axis and all mappings were fitted only on Bearing1_1–Bearing1_3. Bearing1_4 and Bearing1_5 supplied only their own first 15 samples as an online reference. Normalized lifetime is a weak chronological proxy, not measured damage.
