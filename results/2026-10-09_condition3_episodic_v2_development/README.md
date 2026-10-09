# Condition 3 episodic v2 development replay

Fixed development replay of Bearing3_1–Bearing3_3. No Bearing3_4/Bearing3_5 access.

| Artifact | Contents |
|---|---|
| summary.csv | Six bearing/feature summaries |
| episodes.csv | 20 confirmed episodes, exit and censoring records |
| trajectories.csv | 10,810 causal per-step outputs |
| report.json | Configuration hash and implementation checks |
| v1_v2_event_comparison.png | Latched v1 versus resettable v2 event state |
| feature_deviation.png | Standardized feature deviation (symlog axis) |

Parameters: 15 initial measurements, threshold mean+2 population SD,
5 consecutive exceedances to enter, 5 consecutive non-exceedances to exit,
10-step causal trend. Step IDs are zero-based; duration is in measurement steps.

Key result: B3_1 RMS exits the early event at step 34 and confirms a later event
at 2339. B3_2 RMS produces 12 episodes; kurtosis produces none. B3_3 has one
terminal episode per feature. These are vibration-change candidates, not verified
faults; exit does not establish physical recovery. Five terminal episodes are
right-censored. Fewer active steps do not establish fewer false alarms.

Protocol and Chinese interpretation:
../../docs/experiments/episodic_v2_development_protocol_2026-10-09.md
and ../../docs/experiments/episodic_v2_development_result_2026-10-09.md.

Configuration: ../../configs/condition3_episodic_v2_development_20261009.json.
The runner refuses overwriting output or execution records. The numeric completion
record predates plots; plots are reproducible from the frozen CSVs.

Correction (2026-10-09): the total episode count is 19, not 20: 14 completed
and five right-censored episodes. The original CSV records are unchanged and correct.
