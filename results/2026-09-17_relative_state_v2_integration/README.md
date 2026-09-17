# 2026-09-17 Relative State Interpreter v2 integration

This immutable result directory contains the integrated unlabeled relative state `[Level, Trend, Movement]`.

- Level: frozen signed degradation-axis projection with causal smoothing.
- Trend: causal slope of recent Level values.
- Movement: Residual-GRU ensemble prediction surprise divided by the training-bearing median surprise.

All integrity checks passed. Archived Level was reproduced within `1.7e-7`, prefix causality passed, frozen hashes were unchanged, and the training Movement median was `1.00000075`.

## Files

- `experiment_report.json`: hashes, movement reference and integrity decisions.
- `relative_state_trajectories.csv`: per-step Level, Trend and Movement.
- `summary_by_bearing.csv`: descriptive trajectory metrics.
- `integrity_checks.csv`: archived-Level and prefix checks.
- `relative_state_v2.png`: evaluation-bearing state trajectories.

These are relative signals only. They are not absolute damage, RUL or physical health percentages.

