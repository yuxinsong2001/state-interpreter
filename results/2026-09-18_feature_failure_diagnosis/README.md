# Feature LSTM failure diagnosis

Development-only diagnostics on cached B3_1–B3_3 features. B3_4/B3_5 were not read.

Main findings:

- bearing identity accuracy: 90.3%;
- 34/65 features show robust cross-bearing direction conflicts;
- only 1/65 is strong and same-sign on all three bearings;
- all three Ridge LOBO lifetime correlations are negative;
- a 10-step window spans materially different lifetime fractions.

This evidence localizes the bottleneck to representation alignment and task definition rather than Feature LSTM capacity alone.
