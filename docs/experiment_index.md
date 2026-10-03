# Experiment Index

This index groups the main experiments by research question. It is a navigation aid, not a replacement for the original reports and machine-readable records.

## Baseline validation and stability

| Experiment | Date / status | Data split | Method and primary metric | Key result | Evidence |
|---|---|---|---|---|---|
| First locked relative baseline | 2026-08-02 · completed, not project-wide blind | C1: train B1_1–3, validation B1_4, holdout B1_5 | Relative distance; Level–lifetime Spearman rho | B1_5 rho 0.950 over 37 READY states | [`config`](../configs/xjtu_z8_interpreter_v1.json) |
| Multi-seed LOBO stability | 2026-09-12 · completed | C1 and C2; 5 folds/condition; 3 seeds | z=8 autoencoder + relative interpreter; rho | 30-run mean 0.959 ± 0.048 | [`results`](../results/2026-09-12_v0_1_stability_study/README.md) |

## Cross-condition validation

| Experiment | Date / status | Data split | Method and primary metric | Key result | Blind? / evidence |
|---|---|---|---|---|---|
| Arm A / Arm B development | 2026-08-02 · completed | C2 B2_1–3 development, B2_4 validation | frozen source vs target-trained encoder | Arm B especially improved B2_4 in development | No · [`config`](../configs/xjtu_cross_condition_v2_1.json) |
| Bearing2_5 joint evaluation | 2026-09-12 · completed | C2 B2_5 only; 339 measurements | fixed interpreter; Level rho | Arm A 0.946, Arm B 0.964, delta +0.018 | Yes, one-time joint run · [`results`](../results/2026-09-12_bearing2_5_blind/README.md) |

## HMM experiments

| Experiment | Date / status | Data | Method | Key result | Evidence |
|---|---|---|---|---|---|
| Left-right HMM | 2026-09-16 · exploratory | Condition 1 archived embeddings | 3-state Gaussian HMM | reduced short reversals, but overconfident and collapsed on B1_4 | [`results`](../results/2026-09-16_hmm_interpreter_exploratory/README.md) |
| Bearing-centered HMM ablation | 2026-09-16 · exploratory | Condition 1 | centered emissions / calibration ablation | useful diagnostic, not selected as final interpreter | [`results`](../results/2026-09-16_hmm_bearing_centering_ablation/README.md) |

## GRU experiments

| Experiment | Date / status | Data | Method / metric | Key result | Evidence |
|---|---|---|---|---|---|
| Predictive GRU multi-seed | 2026-09-16 · exploratory, verified | Condition 1; 3 seeds | next-z training; hidden-distance Level | stable training, no stable overall advantage over distance Level | [`results`](../results/2026-09-16_gru_predictive_multiseed_stability/README.md) |
| GRU vs persistence | 2026-09-16 · completed comparison | fixed GRU checkpoints | one-step MSE relative to persistence | all 30 fixed comparisons had negative skill | [`results`](../results/2026-09-16_gru_persistence_comparison/README.md) |
| Residual GRU | 2026-09-17 · completed | archived Condition 1 representations | prediction residual for Movement | retained as surprise component, not as absolute health state | [`results`](../results/2026-09-17_residual_gru_termination/README.md) |

## Representation audits

| Experiment | Date / status | Data | Question | Key result | Evidence |
|---|---|---|---|---|---|
| Frozen z=8 audit | 2026-09-17 · completed | 616 Condition 1 embeddings | what information is encoded? | strong shared direction; weak universal stage calibration | [`results`](../results/2026-09-17_z8_representation_audit/README.md) |
| Relative State v2 integration | 2026-09-17 · completed | Condition 1 archived artifacts | can interpretable components be combined causally? | signed Level, causal Trend and residual Movement integrated | [`results`](../results/2026-09-17_relative_state_v2_integration/README.md) |

## Condition 3 domain-shift investigation

| Experiment | Date / status | Data access | Method / gate | Key result | Evidence |
|---|---|---|---|---|---|
| Frozen development diagnostic | 2026-09-18 · failed transfer gate | B3_1–B3_3 only | frozen Relative State v2 | rho -0.391 / +0.894 / -0.564 | [`results`](../results/2026-09-18_condition3_frozen_development/README.md) |
| Reconstruction encoder LOBO | 2026-09-18 · failed consistency gate | B3_1–B3_3 development | retrained z=8 autoencoder, 3 seeds | direction remained unstable | [`checkpoint`](experiments/current_research_checkpoint_2026-09-17.md) |
| Two-sided axial validation | 2026-09-18 · validation failed | B3_4 observed once after lock | post-hoc two-sided Level | validation rho 0.026 | [`results`](../results/2026-09-18_condition3_two_sided_validation/summary.json) |
| Health-aware ranking LOBO | 2026-09-18 · failed all-positive gate | B3_1–B3_3 only | reconstruction + temporal ranking | rho 0.709 / 0.522 / -0.398 | [`results`](../results/2026-09-18_condition3_health_aware_lobo/report.json) |

## Protected boundary

Condition 3 Bearing3_5 remains reserved in the documented protocol and must not be consumed merely to obtain a more favorable result. Development and failed-validation evidence should guide a new hypothesis before any final test is considered.

