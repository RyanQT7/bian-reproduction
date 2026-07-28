import unittest

from bian.data.validators import ValidationError
from bian.methods.enhanced_stage2 import score_stage2, validate_stage2


class EnhancedStage2Tests(unittest.TestCase):
    def test_alias_validation_rejects_duplicates(self):
        item = {
            "candidate_id": "C01",
            "local_anomaly_score": 0.5,
            "temporal_precedence_score": 0.5,
            "topology_upstream_score": 0.5,
            "fault_pattern_compatibility_score": 0.5,
            "symptom_likelihood": 0.2,
            "supporting_evidence_ids": [],
            "counter_evidence_ids": [],
            "reason": "x",
        }
        with self.assertRaises(ValidationError):
            validate_stage2({"candidates": [item, item]}, {"C01", "C02"})

    def test_fixed_scoring_normalizes_top5(self):
        items = []
        aliases = {}
        for index in range(1, 7):
            alias = f"C{index:02d}"
            aliases[alias] = f"region-1-service-{index}"
            items.append(
                {
                    "candidate_id": alias,
                    "local_anomaly_score": index / 10,
                    "temporal_precedence_score": 0.5,
                    "topology_upstream_score": 0.5,
                    "fault_pattern_compatibility_score": 0.5,
                    "symptom_likelihood": 0.1,
                    "supporting_evidence_ids": [],
                    "counter_evidence_ids": [],
                    "reason": "x",
                }
            )
        top5, ranks = score_stage2(
            items,
            aliases,
            {
                "local_anomaly_score": 0.3,
                "temporal_precedence_score": 0.25,
                "topology_upstream_score": 0.25,
                "fault_pattern_compatibility_score": 0.2,
                "symptom_likelihood_penalty": 0.15,
            },
        )
        self.assertEqual(len(top5), 5)
        self.assertAlmostEqual(sum(x["failure_score"] for x in top5), 1)
        self.assertEqual(ranks["rounds"], 3)


if __name__ == "__main__":
    unittest.main()
