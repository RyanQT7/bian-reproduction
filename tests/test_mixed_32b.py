import unittest

from bian.data.validators import ValidationError
from bian.methods.mixed_32b import (
    apply_v3_stage1_fusion,
    stage1_fallback_top5,
    validate_stage2_round,
    validate_type_result,
)


class Mixed32BTests(unittest.TestCase):
    def test_v3_fusion_uses_frozen_component_floors(self):
        item = {
            "candidate_id": "C01",
            "local_anomaly_score": 0.1,
            "temporal_precedence_score": 0.2,
            "topology_upstream_score": 0.3,
            "fault_pattern_compatibility_score": 0.1,
            "symptom_likelihood": 0.1,
            "supporting_evidence_ids": [],
            "counter_evidence_ids": [],
            "concise_reason": "x",
        }
        stage1 = {
            "C01": {
                "deterministic_feature_score": 0.6,
                "model_anomaly_score": 0.4,
                "temporal_change_score": 0.7,
                "direct_fault_evidence_score": 0.8,
                "symptom_likelihood": 0.4,
            }
        }
        fused = apply_v3_stage1_fusion([item], stage1)[0]
        self.assertAlmostEqual(fused["local_anomaly_score"], 0.7)
        self.assertEqual(fused["temporal_precedence_score"], 0.7)
        self.assertEqual(fused["fault_pattern_compatibility_score"], 0.8)
        self.assertEqual(fused["symptom_likelihood"], 0.4)
        self.assertEqual(item["local_anomaly_score"], 0.1)

    def test_stage1_fallback_is_normalized(self):
        ranking = [
            {"node_id": f"region-1-service-{index}", "stage1_score": 6-index}
            for index in range(1, 6)
        ]
        top5 = stage1_fallback_top5(ranking)
        self.assertEqual(len(top5), 5)
        self.assertAlmostEqual(sum(x["failure_score"] for x in top5), 1.0)
        self.assertEqual(top5[0]["node_id"], "region-1-service-1")

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
