# MOMENT no-traffic v2 development experiment

This pipeline excludes every `traffic_*.csv` at source-file level and constructs
device-local series from the explicit region directory and `node` column. It never joins
windows across devices or gaps over 120 seconds. Calibration is fixed at 04:00–05:30 UTC
and cannot generate events.

Continuous and cumulative metrics use MOMENT reconstruction. Sparse counters require an
observed activity departure in addition to reconstruction error. Binary states use only
persisted raw 0/1 state. Device triggers require two distinct metric families, a gated
sparse counter, or a binary departure for two points. All output is labeled a development
experiment.
