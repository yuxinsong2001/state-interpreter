# 2026-09-17 Frozen z=8 Representation Audit

This deterministic audit examines the existing 616 latent observations without retraining the AutoEncoder.

It compares raw and per-bearing early-centred embeddings using PCA geometry, leave-one-bearing-out stage classification, leave-one-bearing-out lifetime regression, bearing-identity prediction, per-dimension correlations, and early-to-late direction cosine.

Main finding: the eight-dimensional representation has an effective dimension of approximately one and a strongly shared degradation direction. Cross-bearing lifetime ordering remains accessible, while absolute stage calibration—especially for Bearing1_4—is weak. The evidence therefore points more strongly to Interpreter/calibration limitations than to a complete absence of degradation information, while also showing that the Encoder representation is too narrow for disentangled health semantics.

Only one operating condition is present, so condition invariance was not tested. See `experiment_report.json` and the protocol/result documents for limitations.
