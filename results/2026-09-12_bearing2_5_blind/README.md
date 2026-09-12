# Bearing2_5 — final joint blind evaluation

Executed: **2026-09-12**. Protocol: `xjtu_cross_condition_v2_1`.

| Frozen pipeline | Level–lifetime Spearman rho |
|---|---:|
| Arm A: source encoder and normalization | 0.9461833064 |
| Arm B: target-trained encoder and normalization | 0.9642422799 |
| B minus A | +0.0180589735 |

339 measurements; 15 calibration measurements and 324 READY states per arm. Both arms share the same input and fixed 15/10 Interpreter parameters.

## Files

- `states.csv`: unchanged joint per-measurement outputs for both arms.
- `state_trajectories.png`: corrected blind-evaluation title, plotted from saved CSV only.
- `execution_report_original.json`: untouched execution output. Its 2026-08-02 date/ID is a legacy script metadata error; the actual run date is 2026-09-12.
- `evaluation_record.json`: actual run date, frozen input hashes, metrics and interpretation.
- `summary_zh.md`: Chinese result explanation and limitations.
- `SHA256SUMS.txt`: checksums of delivery files other than the checksum file itself.

Original local artifacts remain in `runs/joint_blind_bearing2_5_20260912/`. The archived original plot there has the obsolete rehearsal title. No raw vibration data or model checkpoints are included here.

## Interpretation and handover boundary

Both pipelines exceed the preregistered rho threshold of 0.7 on this held-out bearing. Arm B has a small descriptive improvement; this is not a statistical significance claim. One bearing does not establish broad generalization, physical damage calibration, or usefulness for RL decisions. Different fitted input normalization makes reconstruction MSE unsuitable as the main between-arm comparison.

The blind evaluation is completed. Bearing2_5 must not be reused for parameter or checkpoint selection. Configuration and checkpoints were not changed after evaluation. The post-evaluation script change adds a record-based rerun guard and fixes plot/date metadata; it does not change the inference calculations used for the result. Existing configuration flags are historical preregistration fields; the completed record is authoritative for execution status.

## Actual Interpreter implementation

The reference is the mean of the first 15 embeddings. Level is the trailing mean of Euclidean distances to that reference, without division by a calibration standard deviation. Trend is the least-squares slope over recent Level values. Movement is the distance between consecutive embeddings, without temporal averaging. Earlier presentation formulas suggesting otherwise should not be used as implementation documentation.

Validation recorded at closeout: 66 tests passed; a second blind invocation was rejected before data loading. Git does not contain the large dataset/checkpoints, so full inference reproduction requires those local artifacts and their recorded hashes.
