import unittest

from bian.data.enhanced_features import (
    extract_device_evidence,
    source_semantic_status,
    stable_change,
)


class EnhancedFeatureTests(unittest.TestCase):
    def source(self, status="available", pre=1, fault=2, post=1):
        return {
            "status": status,
            "applicable": status != "unavailable_by_role",
            "phases": {
                "pre_fault": {
                    "row_count": 1,
                    "first_timestamp_utc": "2026-01-01T00:00:00Z",
                    "last_timestamp_utc": "2026-01-01T00:00:00Z",
                    "metrics": {"bgp_peer_up": {"mean": pre}},
                },
                "fault": {
                    "row_count": 1,
                    "first_timestamp_utc": "2026-01-01T00:01:00Z",
                    "last_timestamp_utc": "2026-01-01T00:01:00Z",
                    "metrics": {"bgp_peer_up": {"mean": fault, "max": fault, "min": fault}},
                },
                "post_fault": {
                    "row_count": 1,
                    "first_timestamp_utc": "2026-01-01T00:02:00Z",
                    "last_timestamp_utc": "2026-01-01T00:02:00Z",
                    "metrics": {"bgp_peer_up": {"mean": post}},
                },
            },
        }

    def test_zero_baseline_is_bounded(self):
        self.assertLessEqual(stable_change(0, 1000000), 1)

    def test_unavailable_is_neutral(self):
        device = {
            "node_id": "region-1-service-1",
            "device_family": "service",
            "sources": {"routing_metrics": self.source("unavailable_by_role")},
        }
        result = extract_device_evidence(device)
        self.assertEqual(result["evidence"], [])
        self.assertEqual(
            result["feature_summary"]["unavailable_by_role_anomaly_contribution"], 0
        )

    def test_sudden_missing_differs_from_unavailable(self):
        source = self.source()
        source["phases"]["fault"]["row_count"] = 0
        self.assertEqual(
            source_semantic_status(source), "sudden_missing_during_fault"
        )

    def test_protocol_metric_is_direct_evidence_for_br(self):
        device = {
            "node_id": "region-1-br-1",
            "device_family": "br",
            "sources": {"routing_metrics": self.source()},
        }
        result = extract_device_evidence(device)
        self.assertTrue(result["evidence"][0]["direct_fault_evidence"])
        self.assertIsNotNone(result["evidence"][0]["first_change_time"])


if __name__ == "__main__":
    unittest.main()
