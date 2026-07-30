# MOMENT screened v3 development experiment

This conservative pipeline screens every device/metric independently, excludes all
traffic CSV sources, constants, unreliable near-constants, ambiguous counters, and exact
duplicates. Confirmed counters use per-segment per-minute differences and `log1p`.
Sparse counters are audit-only evidence and cannot trigger. Device events require two
distinct metric families or a confirmed raw binary-state departure for two samples.

Score thresholds are fixed before inference: high is max(P99.9, median + 8 robust scales);
low is max(P99.5, median + 5 robust scales). Calibration never produces events.
