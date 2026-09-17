# State Interpreter research checkpoint — 2026-09-17

## Completed

- z=8 AutoEncoder latent extraction and audit.
- Euclidean relative Level baseline.
- HMM and direct-GRU investigations.
- Persistence comparison and Residual-GRU correction.
- Training-only signed degradation-axis Level.
- Early-window scale calibration rejection.
- Integrated Relative State Interpreter v2: signed Level, causal Trend, Residual-GRU Movement.
- Full repository test status: 121 passed.

## Current scientific conclusion

The latent representation contains a shared degradation-related direction and supports strong within-bearing relative ordering. A unified cross-bearing absolute health scale has not been established. Early-window noise scaling is not suitable. The supported output is therefore a relative state, not a damage percentage.

## Frozen artifacts

- `results/2026-09-17_signed_axis_interpreter/axis.json`
- `results/2026-09-17_residual_gru_termination/residual_seed_*_checkpoint.pt`
- `results/2026-09-17_relative_state_v2_integration/`
- `configs/xjtu_relative_state_v2_integration.json`

## Next executable task

Design a training-only Movement event threshold and evaluate peak persistence, agreement across Residual-GRU seeds, and temporal relation to Level/Trend turning points. Do not call detected peaks faults without event labels.

## Resume command

```powershell
cd "D:\1\德国留学\斯图加特大学在校资料\hiwi工作\state-interpreter"
D:\python3.13\python.exe -m pytest -q -p no:cacheprovider --basetemp "D:\1\德国留学\斯图加特大学在校资料\hiwi工作\student_handover 2\.pytest_resume_relative_v2"
```

