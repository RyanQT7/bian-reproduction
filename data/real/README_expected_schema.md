# Expected real-data contract

Real data must not be committed. A future adapter should map source records to:

- incident: ID, start time, candidate device IDs, alerts, topology, optional
  ground truth, and metadata;
- alert: ID, involved device IDs, source type, start/end time, severity,
  human-readable text, and source-specific attributes;
- topology: nodes, edges, device groups, and directionality;
- ground truth: one or more root device IDs and confirmation provenance.

Required validation includes unique IDs, ISO-8601 timestamps, known candidate
devices, topology edge endpoints, finite non-negative scores, and explicit
ground-truth presence before metric calculation.

The paper does not publish a complete schema, all SOPs, the complete mapping
from data sources to anomaly scenarios, or its private incident dataset.
