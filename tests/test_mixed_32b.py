import unittest

from bian.data.validators import ValidationError
from bian.methods.mixed_32b import validate_stage2_round, validate_type_result


class Mixed32BTests(unittest.TestCase):
    def test_all_zero_stage2_is_rejected(self):
        item = {
            "candidate_id": "C01",
            "local_anomaly_score": 0,
            "temporal_precedence_score": 0,
            "topology_upstream_score": 0,
            "fault_pattern_compatibility_score": 0,
            "symptom_likelihood": 0,
            "supporting_evidence_ids": [],
            "counter_evidence_ids": [],
            "concise_reason": "none",
        }
        with self.assertRaises(ValidationError):
            validate_stage2_round({"candidates": [item]}, {"C01"}, set())

    def test_all_zero_single_type_is_valid_negative_evidence(self):
        item = {
            "candidate_id": "C01",
            "root_role_compatibility": 0,
            "metric_pattern_compatibility": 0,
            "protocol_state_compatibility": 0,
            "temporal_pattern_compatibility": 0,
            "topology_context_compatibility": 0,
            "counter_evidence_penalty": 0,
            "supporting_evidence_ids": [],
            "counter_evidence_ids": [],
        }
        result = validate_type_result(
            {"type_id": "T01", "root_hypotheses": [item]},
            "T01",
            {"C01"},
            set(),
        )
        self.assertEqual(result["root_hypotheses"][0]["candidate_id"], "C01")


if __name__ == "__main__":
    unittest.main()
