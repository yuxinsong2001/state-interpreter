# Paired Source-only versus Source-DANN development result

This directory contains the immutable outputs of the preregistered Condition 3 LOBO development run.

- `experiment_report.json`: aggregate results and gate decisions.
- `per_seed_health_metrics.csv`: health ordering for every arm, fold, and seed.
- `health_summary_by_arm_and_bearing.csv`: per-bearing aggregates.
- `paired_arm_comparison.csv`: Source-DANN minus Source-only comparison.
- `identity_probe.csv`: five-block bearing identity probes.
- `health_trajectories.csv`: causal health predictions.
- `training_history.csv`: fixed 30-epoch training histories.
- `*.pt`: 18 final-epoch checkpoints.

The Source-DANN arm improved the mean health correlation and reduced average identity accuracy, but neither preregistered gate passed. Validation on `Bearing3_4` is not authorized; `Bearing3_4` and `Bearing3_5` were not read.

