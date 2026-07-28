import unittest

from bian.methods.enhanced_stage1 import rank_stage1


class EnhancedStage1Tests(unittest.TestCase):
    def test_symptom_penalty_does_not_hard_exclude(self):
        evidence = []
        analyses = []
        for node, role, direct, symptom in (
            ("region-1-br-1", "br", 0.8, 0.05),
            ("region-1-service-1", "service", 0.0, 1.0),
        ):
            evidence.append(
                {
                    "node_id": node,
                    "device_role": role,
                    "evidence": [],
                    "feature_summary": {
                        "deterministic_feature_score": direct,
                        "temporal_change_score": 1.0,
                        "direct_fault_evidence_score": direct,
                        "symptom_likelihood": symptom,
                        "data_quality_adjustment": 0,
                        "earliest_change_time": None,
                    },
                }
            )
            analyses.append({"node_id": node, "anomaly_score": 1.0})
        config = {
            "weights": {
                "deterministic_feature_score": 0.35,
                "model_anomaly_score": 0.1,
                "temporal_change_score": 0.15,
                "direct_fault_evidence_score": 0.35,
                "downstream_region_symptom_score": 0.2,
                "data_quality_adjustment": 1.0,
                "symptom_likelihood_penalty": 0.2,
            },
            "top_p": 0.9,
            "min_candidates": 1,
            "max_candidates": 2,
        }
        ranking, _ = rank_stage1(evidence, analyses, config)
        self.assertEqual(ranking[0]["node_id"], "region-1-br-1")
        self.assertGreaterEqual(ranking[1]["stage1_score"], 0)


if __name__ == "__main__":
    unittest.main()
