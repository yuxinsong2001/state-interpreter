# Relative event state interface verification

This is an interface replay on archived development CSVs, not a new health experiment.
All 10,810 original detector observations match exactly. The 19 termination records
match original episode metadata (14 exits, five end-of-record censored episodes).

- `states.csv`: per-step original output plus interpretation metadata.
- `episode_terminations.csv`: complete or censored episode termination records.
- `example_state_messages.json`: B3_1 RMS messages for entry, exit and re-entry.
- `report.json`: input/code hashes and compatibility checks.

See `../../docs/experiments/relative_event_state_interface_2026-10-09.md`.
The new interface keeps feature deviation, causal trend and event state distinct.
Event exit does not establish physical recovery. No protected bearing was read.
