# 2026-09-17 Residual GRU Termination Experiment

This directory contains the predeclared comparison of frozen direct GRU forecasts, persistence, and three newly trained persistence-centred residual GRUs.

The residual model uses `forecast[t] = z[t] + delta_gru[t]`. Its delta head is zero-initialised, making the untrained forecast exactly equal to persistence.

The predeclared gate passed: all three seeds obtained positive MSE skill on Bearing1_4 and Bearing1_5. Across all five bearings, all 15 residual seed-bearing comparisons had positive MSE skill. This supports further validation, not deployment or a physical-health interpretation. Median-step and win-fraction diagnostics show that the advantage is not uniform across typical time steps.

See `experiment_report.json`, `verification_record.json`, and the protocol/result documents for scope and limitations.
