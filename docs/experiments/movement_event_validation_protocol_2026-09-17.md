# Movement event validation protocol (2026-09-17)

The ensemble event threshold is the pooled 99th percentile of Movement on Bearing1_1–Bearing1_3 from step 25. Each Residual-GRU seed receives its own training-only 99th-percentile surprise threshold. Evaluation events are not used to tune thresholds.

Report event count, persistent runs of at least two consecutive samples, maximum duration, three-seed agreement, and distance to the nearest high-|Trend| point. A detected event is a change candidate, not a labelled fault.
