# State Interpreter research checkpoint — 2026-09-17

## Completed

- z=8 AutoEncoder latent extraction and audit.
- Euclidean relative Level baseline.
- HMM and direct-GRU investigations.
- Persistence comparison and Residual-GRU correction.
- Training-only signed degradation-axis Level.
- Early-window scale calibration rejection.
- Integrated Relative State Interpreter v2: signed Level, causal Trend, Residual-GRU Movement.
- Training-only Movement event threshold validation.
- Condition 3 frozen-v2 preregistration, artifact hash preflight and staged access guards; no Bearing3 data read.
- Full repository test status: 121 passed.

## Current scientific conclusion

The latent representation contains a shared degradation-related direction and supports strong within-bearing relative ordering. A unified cross-bearing absolute health scale has not been established. Early-window noise scaling is not suitable. The supported output is therefore a relative state, not a damage percentage.

## Frozen artifacts

- `results/2026-09-17_signed_axis_interpreter/axis.json`
- `results/2026-09-17_residual_gru_termination/residual_seed_*_checkpoint.pt`
- `results/2026-09-17_relative_state_v2_integration/`
- `configs/xjtu_relative_state_v2_integration.json`

## Next executable task

Condition 3 preflight is complete without reading Bearing3 data. The next authorized task is a one-time development diagnostic on Bearing3_1–Bearing3_3 using token `EXECUTE_BEARING3_DEVELOPMENT_ONCE`; Bearing3_4 and Bearing3_5 remain protected.

## Completed checkpoint — 2026-09-18

The authorized Bearing3_1–Bearing3_3 frozen development diagnostic completed in 1013.95 seconds. Bearing3_4 and Bearing3_5 were not read. The frozen route did not generalize consistently: Level–lifetime Spearman correlations were -0.391, +0.894 and -0.564 for B3_1–B3_3, while B3_2 produced a 13.962% Movement event fraction under the frozen Condition 1 threshold. Full repository tests pass: 126 passed.

The next task is not to consume protected data. Use only B3_1–B3_3 to compare controlled adaptation candidates, freeze one candidate, and then use B3_4 for validation. Keep B3_5 for the one-time final test.

## Model-localization checkpoint — 2026-09-18

Arm A interpreter-only LOBO failed to change the signed-Level result. Arm B retrained reconstruction AutoEncoders across three seeds: B3_1 was sign-unstable (mean rho 0.068), B3_2 stayed positive (0.905), and B3_3 stayed negative across all seeds (-0.557). Encoder adaptation therefore failed its preregistered consistency gate.

A post-hoc two-sided axial Level looked positive on B3_1–B3_3, so its formula and artifact hashes were locked before first access to B3_4. Validation did not support it: B3_4 signed rho was 0.328 and two-sided rho was 0.026. Bearing3_5 remains unread. Do not run the blind test. The next research task is a development-only health-aware or nonlinear/phase-aware representation; B3_4 must now be treated as previously observed validation evidence.

## Resume command

```powershell
cd "D:\1\德国留学\斯图加特大学在校资料\hiwi工作\state-interpreter"
D:\python3.13\python.exe -m pytest -q -p no:cacheprovider --basetemp "D:\1\德国留学\斯图加特大学在校资料\hiwi工作\student_handover 2\.pytest_resume_relative_v2"
```
