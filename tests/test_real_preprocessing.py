from datetime import timezone
import unittest

from bian.data.real_preprocessing import (
    NumericAccumulator,
    assert_no_leakage,
    count_csv_records,
    node_id_for_row,
    parse_utc,
    phase_for,
    validate_experiment_incidents,
)
from bian.data.validators import ValidationError


class RealPreprocessingTests(unittest.TestCase):
    incident = {
        "incident_id": "incident-0001",
        "dataset_timezone": "UTC",
        "fault_start_time_utc": "2026-07-27T06:05:56Z",
        "fault_end_time_utc": "2026-07-27T06:06:51Z",
        "slice_start_time_utc": "2026-07-27T06:00:56Z",
        "slice_end_time_utc": "2026-07-27T06:11:51Z",
        "ground_truth_included": False,
    }

    def test_parse_utc_naive_and_z(self):
        self.assertEqual(parse_utc("2026-07-27 06:00:00").tzinfo, timezone.utc)
        self.assertEqual(parse_utc("2026-07-27T06:00:00Z").tzinfo, timezone.utc)

    def test_phase_boundaries_are_inclusive(self):
        self.assertEqual(phase_for(parse_utc("2026-07-27T06:00:56Z"), self.incident), "pre_fault")
        self.assertEqual(phase_for(parse_utc("2026-07-27T06:05:56Z"), self.incident), "fault")
        self.assertEqual(phase_for(parse_utc("2026-07-27T06:06:51Z"), self.incident), "fault")
        self.assertEqual(phase_for(parse_utc("2026-07-27T06:11:51Z"), self.incident), "post_fault")

    def test_exact_ten_incidents_required(self):
        with self.assertRaises(ValidationError):
            validate_experiment_incidents([self.incident])

    def test_numeric_aggregation_reports_missing(self):
        accumulator = NumericAccumulator()
        for value in ("1", "", "3"):
            accumulator.add(value)
        rendered = accumulator.render()
        self.assertEqual(rendered["count"], 2)
        self.assertEqual(rendered["missing_count"], 1)
        self.assertEqual(rendered["mean"], 2)
        self.assertEqual(rendered["delta"], 2)

    def test_leakage_keys_rejected(self):
        with self.assertRaises(ValidationError):
            assert_no_leakage({"nested": {"fault_type": "secret"}}, {"fault_type"})

    def test_candidate_role_mapping_excludes_monitor(self):
        self.assertEqual(
            node_id_for_row("node_metrics", {"node": "service-vm-2"}, "region-3"),
            "region-3-service-2",
        )
        self.assertIsNone(
            node_id_for_row("node_metrics", {"node": "monitor-vm"}, "region-3")
        )

    def test_count_csv_records(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            self.assertEqual(count_csv_records(path), 2)

    def test_optional_topology_protocol_can_be_null(self):
        edge = {"source": "a", "target": "b", "edge_type": "physical"}
        rendered = {
            "source": edge["source"],
            "target": edge["target"],
            "edge_type": edge["edge_type"],
            "protocol": edge.get("protocol"),
        }
        self.assertIsNone(rendered["protocol"])

    def test_device_phase_gap_is_not_dataset_coverage_failure(self):
        phase_counts = {"pre_fault": 2, "fault": 0, "post_fault": 2}
        status = (
            "empty"
            if sum(phase_counts.values()) == 0
            else "partial"
            if any(value == 0 for value in phase_counts.values())
            else "available"
        )
        self.assertEqual(status, "partial")


if __name__ == "__main__":
    unittest.main()
