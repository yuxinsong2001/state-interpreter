# GRU temporal-ranking objective experiment

Six paired runs compare prediction-only GRU training with the same GRU plus a fixed temporal-ranking penalty. Three seeds, identical initializations, 300 epochs, no hyperparameter sweep. All metrics start at step 25.

The ranking objective did not meet the preregistered stability gate. Bearing1_4 seed standard deviation increased from 0.06321 to 0.06527; Bearing1_5 decreased from 0.03197 to 0.02659 (16.82%, below the 20% gate). Neither arm establishes physical health semantics from time correlation.

## Files

- `experiment_report.json`: configuration, provenance, environment, hashes, summaries, decisions.
- `summary_by_bearing.csv`: mean/std/range across seeds plus the distance baseline.
- `decision_checks.csv`: predeclared gate checks, both bearings fail the overall gate.
- `per_seed_bearing_metrics.csv`: 35 metric rows (30 GRU + 5 distance baseline).
- `per_step_levels.csv`: 3,696 rows for the six GRU runs, including distance baseline values.
- `training_history.csv`: 1,800 epoch rows.
- `training_summary.csv`: six selected checkpoints, all epoch 300.
- `control_seed_*_checkpoint.pt`, `ranking_seed_*_checkpoint.pt`: six frozen models.
- `negative_control_metrics.csv`: 60 constant/shuffled-input diagnostic rows.
- `reproduction_check.json`: exact reproduction of three earlier control checkpoints.
- `level_comparison.png`: all seed trajectories; rescaling is for display only.
- `execution_record.json`, `recovery_record.json`: execution status and honest record of the matplotlib failure and report recovery, without retraining.

Detailed Chinese interpretation: [experiment report](../../docs/experiments/gru_temporal_ranking_result_2026-09-16.md).

## Execution

Install the project's analysis dependencies before training. Use an explicit Python interpreter when multiple installations are present. The executed training script is retained unchanged for provenance.

```powershell
python -B scripts/run_gru_temporal_ranking.py --config configs/xjtu_gru_temporal_ranking_v1.json
```

An existing output directory is never overwritten by this command. `finalize_gru_temporal_ranking.py` is the one-time recovery path for this saved run; it refuses to overwrite final artifacts and verifies every saved Level and prediction MSE against frozen checkpoint inference before completing the report.

