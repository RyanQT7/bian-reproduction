import unittest

from bian.predictions import validate_predictions
from scripts.run_blind33_vllm import normalize_taxonomy


class Blind33Tests(unittest.TestCase):
    def test_taxonomy_requires_exact_32_ordered_ids(self):
        candidates = [
            {
                "candidate_index": index,
                "fault_type": f"type_{index}",
                "category": ("link", "firewall", "resource", "route", "service")[
                    (index - 1) % 5
                ],
                "description_zh": "public definition",
                "allowed_variants": [],
            }
            for index in range(1, 33)
        ]
        document = {
            "candidate_count": 32,
            "categories": ["link", "firewall", "resource", "route", "service"],
            "candidates": candidates,
        }
        taxonomy = normalize_taxonomy(
            document, [item["fault_type"] for item in candidates]
        )
        self.assertEqual(len(taxonomy), 32)
        self.assertEqual(taxonomy[-1]["type_id"], "T32")

    def test_explicit_stage1_fallback_prediction_validates(self):
        nodes = [f"region-1-service-{index}" for index in range(1, 6)]
        taxonomy = [
            {"fault_type": f"type_{index}", "fault_category": "service"}
            for index in range(1, 4)
        ]
        top3 = [
            {
                "rank": index,
                "fault_type": item["fault_type"],
                "fault_category": item["fault_category"],
                "confidence": 1 / 3,
            }
            for index, item in enumerate(taxonomy, 1)
        ]
        record = {
            "incident_id": "incident-0001",
            "prediction_status": "success",
            "classification_status": "success",
            "prediction_mode": "stage1_fallback_after_stage2_failure",
            "top5_root_causes": [
                {"rank": index, "node_id": node, "failure_score": 0.2}
                for index, node in enumerate(nodes, 1)
            ],
            "predicted_fault_type": top3[0]["fault_type"],
            "predicted_fault_category": top3[0]["fault_category"],
            "fault_type_top3": top3,
            "rank_of_ranks": {
                "rounds": 0,
                "raw_rankings": [],
                "fallback": "stage1",
            },
        }
        result = validate_predictions(
            [record],
            expected_incident_ids={"incident-0001"},
            candidate_node_ids=set(nodes),
            taxonomy=taxonomy,
            minimum_rank_rounds=2,
        )
        self.assertTrue(result["valid"], result["errors"])


if __name__ == "__main__":
    unittest.main()
