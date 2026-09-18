# Health-aware Encoder checkpoint — 2026-09-18

## Motivation

Condition 3 evidence rejected interpreter-only recalibration, ordinary reconstruction-AutoEncoder adaptation, and a post-hoc two-sided axial Level. Bearing3_5 remains unread. Bearing3_4 has already been observed and may only be used as historical evidence, not repeated model selection.

## Next experiment

Train a reconstruction AutoEncoder with an additional within-bearing temporal-ordering head. The ordering loss uses pairs from the same training bearing only and requires later measurements to receive a larger scalar health score. Evaluation is LOBO on Bearing3_1–Bearing3_3.

```text
STFT → Encoder → z=8 ─┬→ Decoder reconstruction
                       └→ linear health head → temporal ranking loss
```

Fixed first protocol: five epochs, seed 20260918, batch 32, learning rate 1e-3, reconstruction weight 1.0, ranking weight 0.1, minimum normalized-lifetime separation 0.1. Each fold fits normalization, AutoEncoder and health head only on its two training bearings. The held-out bearing is used once for evaluation and never for checkpoint selection.

## Decision gate

Continue this family only if held-out health-score Spearman is positive on all three development bearings. Otherwise stop tuning the linear health head and move to nonlinear/phase-aware representation. Do not read Bearing3_5.

## Resume point

Implement `configs/xjtu_condition3_health_aware_lobo.json` and `scripts/run_condition3_health_aware_lobo.py`, then run the three folds and perform the 11-item validity scan. Sync results to `work_log.md` and `presentation_notes.md`.

## Completed result

The fixed first protocol completed. Held-out Spearman rho was 0.709 for B3_1, 0.522 for B3_2 and -0.398 for B3_3. The temporal-ranking head substantially improved B3_1 and changed B3_2, but B3_3 remained negative; therefore the preregistered all-positive gate failed. Stop tuning the linear health head. The result supports moving to a nonlinear or phase-aware interpreter on development data. Bearing3_5 remains unread.
