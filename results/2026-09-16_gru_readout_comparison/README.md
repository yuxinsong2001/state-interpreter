# GRU State Readout Comparison

This directory contains a fixed-checkpoint ablation of three causal State Interpreter readouts. No GRU was retrained in this experiment.

## Compared readouts

- `hidden_distance`: distance from the current hidden state to its early reference;
- `predicted_z_distance`: distance from the causal one-step embedding forecast to its early reference;
- `prediction_residual`: distance between the observed embedding and the forecast made at the preceding step.

All reported metrics use a common scoring start at step 25.

## Main conclusion

Moving the distance readout from hidden space to predicted-z space did not materially reduce seed sensitivity on Bearing1_4 and Bearing1_5. The prediction residual is not a monotonic health level, but remains a candidate auxiliary change or anomaly signal.

## Files

- `experiment_report.json`: machine-readable summary, hashes, warnings, and aggregate results;
- `per_step_readouts.csv`: all aligned per-step readout values;
- `per_seed_bearing_metrics.csv`: metrics for each seed, bearing, and readout;
- `summary_by_readout_bearing.csv`: across-seed bearing summaries;
- `trajectory_consistency.csv`: pairwise cross-seed trajectory correlations;
- `readout_aggregate.csv`: descriptive aggregate comparison.

See `../../docs/experiments/gru_readout_comparison_result_2026-09-16.md` for interpretation and limitations.
