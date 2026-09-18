# Feature LSTM Condition 3 development training v2

This directory contains the preregistered 3-fold LOBO × 3-seed development run.

- Input: cached 65-feature sequences from B3_1–B3_3
- Scaling: causal early z-score followed by signed log1p
- Model: 11,169-parameter Feature BiLSTM
- Training: fixed 50 epochs, final checkpoint only
- Validation gate: failed
- Protected bearings: B3_4 and B3_5 were not read

Primary files:

- `experiment_report.json`: aggregate decision and protocol integrity
- `per_seed_bearing_metrics.csv`: nine LOBO results
- `summary_by_bearing.csv`: bearing-level mean and standard deviation
- `training_history.csv`: fixed-epoch training curves
- `trajectories.csv`: held-out development trajectories
- `seed_*_final.pt`: final epoch checkpoints

The failed development gate prohibits evaluation on B3_4 under this protocol.
