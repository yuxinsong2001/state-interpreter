# 2026-09-17 Frozen GRU Increment Diagnosis

This directory contains a read-only diagnostic evaluation of six frozen GRU checkpoints.

The experiment compares observed increments `z[t+1]-z[t]` with predicted increments `forecast[t]-z[t]` for targets starting at step 25. It diagnoses mean bias, direction and magnitude without retraining or selecting checkpoints.

Main conclusion: constant bias is not the dominant failure mode. Oracle mean-bias correction does not beat persistence in any of the 30 arm/seed/bearing comparisons. Direction and magnitude are bearing-dependent, and the temporal-ranking arm is almost identical to the control arm.

See `experiment_report.json` for the immutable configuration, hashes, limitations and aggregated metrics. The oracle bias correction is post-hoc and not deployable.
