# Real-data RCA experiment scope

## BiAn starts after incident detection

BiAn does not detect whether an incident occurred or estimate its time range.
Section 3 and Figure 4 of the paper state that the system is invoked “when an
incident is reported” and receives monitoring alerts for candidate devices.
Section 2.2 and Figure 1 place anomaly detection and self-healing upstream of
operator investigation. Pipeline 1 single-device anomaly analysis identifies
device symptoms inside a known incident; it is not incident detection.

This experiment therefore receives incident IDs and UTC intervals from an
upstream system and evaluates only:

- root-cause node localization;
- fault-type classification.

It does not calculate detection, time-IoU, event matching, detection FP/FN, or
detection scores.

## Leakage boundary

Preprocessing and prediction may read only the experiment metadata, topology,
schemas, and raw monitoring sources. Evaluation labels are unavailable until a
validated prediction file has been hashed and frozen.

Region IDs in machine-readable inputs are `region-1` through `region-8`.
Candidate IDs follow `region-{index}-{role}`. Source location names are used
only to locate raw directories and are never included in model inputs.

## Uniform preprocessing

The preprocessing configuration is shared across all regions and devices.
Per-device records contain the same source keys. A source that does not apply
to a role or has no records is represented explicitly rather than omitted.
Dataset-window completeness is checked at the region/source-file level.
Device-level gaps inside an otherwise covered window are retained as `partial`
or `empty`, because a telemetry interruption can itself be RCA evidence.
High-volume raw five-tuple records are excluded uniformly; already aggregated
traffic-flow metrics remain available.

All rules in the committed configuration are implementation choices unless
explicitly marked otherwise.
