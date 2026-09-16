# Frozen GRU versus persistence

This evaluation compares six fixed checkpoints (prediction-only and temporal-ranking, three seeds each) against the one-step persistence forecast. No fitting or checkpoint selection occurs.

All 30 primary seed/bearing/arm comparisons have negative skill. Mean GRU-to-persistence MSE ratios across seeds range from approximately 1.52 to 4.53 across bearings/arms. Current GRUs have not demonstrated additional one-step predictive skill.

## Scoring

Forecast origin `t` predicts target `t+1`. Primary target steps are 25 through the final observation; the same history, scaling, targets and feature dimensions are used for both methods. `skill = 1 - MSE_GRU / MSE_persistence`; larger is better. The full-sequence diagnostic scope reproduces historical MSE and is not pre-calibration online performance evidence.

## Artifacts

- `experiment_report.json`: configuration, hashes, runtime, checks, summaries, limits.
- `summary_by_bearing.csv`: 20 rows; filter `scope=primary` for main results.
- `per_seed_bearing_metrics.csv`: 60 rows, including the full-sequence diagnostic scope.
- `per_target_errors.csv`: 3,666 aligned error pairs and primary-score membership.
- `integrity_checks.csv`: 30 exact archived-MSE, prefix-causality and baseline checks.
- `prediction_skill.png`: MSE ratios per seed/bearing; the dashed line at 1 is persistence.

Detailed interpretation: [Chinese experiment report](../../docs/experiments/gru_persistence_comparison_result_2026-09-16.md).

All bearings were previously inspected. Seed variability is not between-device uncertainty, and forecasting skill does not establish physical health semantics. The output directory is immutable; the script rejects an existing destination.
