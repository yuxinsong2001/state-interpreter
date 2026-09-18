# TS2Vec Condition 3 development result

This directory contains the immutable outputs of the preregistered compact causal TS2Vec development run.

- `experiment_report.json`: aggregate gates and protocol status.
- `per_seed_health_metrics.csv`: LOBO health metrics for every seed.
- `health_summary_by_bearing.csv`: aggregate health results.
- `identity_probe.csv`: bearing identity probe results.
- `health_trajectories.csv`: causal health trajectories.
- `training_history.csv`: fixed-iteration training history.
- `*.pt`: the nine fold/seed checkpoints.

The combined gate failed. `Bearing3_4` and `Bearing3_5` were not read, and validation is not authorized. See `docs/experiments/xjtu_ts2vec_development_result_2026-09-18.md` for interpretation.

