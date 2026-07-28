import unittest

from bian.data.validators import ValidationError
from bian.real_inference import (
    aggregate_stage2_rounds,
    compact_device,
    cumulative_top_p,
)


class RealInferenceTests(unittest.TestCase):
    def test_compaction_is_deterministic_and_preserves_partial(self):
        source = {
            "status": "partial",
            "missing_phases": ["post_fault"],
            "phases": {
                "pre_fault": {"metrics": {"cpu": {"mean": 1.0}}},
                "fault": {"metrics": {"cpu": {"mean": 3.0}}},
                "post_fault": {"metrics": {"cpu": {"mean": None}}},
            },
        }
        device = {
            "node_id": "region-1-br-1",
            "device_family": "br",
            "sources": {"routing_metrics": source},
        }
        first = compact_device(device, 5)
        self.assertEqual(first, compact_device(device, 5))
        self.assertEqual(first["source_states"]["routing_metrics"]["status"], "partial")

    def test_cumulative_top_p_is_bounded_and_validated(self):
        candidates = tuple(f"n-{index}" for index in range(10))
        scores = {node: 0.01 for node in candidates}
        scores[candidates[0]] = 0.91
        kept = cumulative_top_p(scores, candidates, 0.8, 6)
        self.assertEqual(len(kept), 5)
        with self.assertRaises(ValidationError):
            cumulative_top_p(scores, candidates, 0, 6)

    def test_rank_of_ranks_produces_normalized_top5(self):
        candidates = tuple(f"region-1-service-{i}" for i in range(1, 6))
        taxonomy = tuple(
            {"fault_type": f"t{i}", "fault_category": f"c{i}"} for i in range(3)
        )
        roots = [
            {"node_id": node, "score": 1, "reason": node} for node in candidates
        ]
        faults = [
            {
                "fault_type": item["fault_type"],
                "fault_category": item["fault_category"],
                "confidence": 1,
                "reason": "r",
            }
            for item in taxonomy
        ]
        top5, top3, rankings = aggregate_stage2_rounds(
            [{"root_causes": roots, "fault_types": faults, "analysis": "a"}] * 3,
            candidates,
            taxonomy,
        )
        self.assertAlmostEqual(sum(item["failure_score"] for item in top5), 1.0)
        self.assertEqual(len(top3), 3)
        self.assertEqual(len(rankings), 3)


if __name__ == "__main__":
    unittest.main()
