import unittest

from bian.data.validators import ValidationError
from bian.evaluation.real_scoring import score_predictions


class RealScoringTests(unittest.TestCase):
    def truth(self, index):
        return {
            "incident_id": f"incident-{index:04d}",
            "root_node": f"region-{index}-br-1",
            "fault_type": "exact" if index % 2 else "other",
            "fault_category": "routing",
        }

    def prediction(self, index, hit_rank=1, fault_type=None, category="routing"):
        truth_node = f"region-{index}-br-1"
        nodes = [
            f"region-{index}-br-2",
            f"region-{index}-cr-1",
            f"region-{index}-cr-2",
            f"region-{index}-fw",
            f"region-{index}-service-1",
        ]
        nodes.insert(hit_rank - 1, truth_node)
        nodes = nodes[:5]
        return {
            "incident_id": f"incident-{index:04d}",
            "top5_root_causes": [
                {"rank": rank, "node_id": node, "failure_score": 0.2}
                for rank, node in enumerate(nodes, 1)
            ],
            "predicted_fault_type": fault_type or ("exact" if index % 2 else "other"),
            "predicted_fault_category": category,
            "fault_type_top3": [
                {"rank": 1, "fault_type": fault_type or ("exact" if index % 2 else "other")},
                {"rank": 2, "fault_type": "backup"},
                {"rank": 3, "fault_type": "third"},
            ],
        }

    def score(self, predictions, truths, count):
        return score_predictions(
            predictions=predictions,
            ground_truth=truths,
            root_node_field="root_node",
            expected_case_count=count,
        )

    def test_perfect_score_is_70(self):
        truths = [self.truth(index) for index in range(1, 4)]
        predictions = [self.prediction(index) for index in range(1, 4)]
        result = self.score(predictions, truths, 3)
        self.assertEqual(result["localization"]["score_40"], 40)
        self.assertEqual(result["classification"]["score_30"], 30)
        self.assertEqual(result["total_score_70"], 70)

    def test_rank_and_category_partial_credit(self):
        truth = [self.truth(1)]
        prediction = [self.prediction(1, hit_rank=3, fault_type="wrong")]
        result = self.score(prediction, truth, 1)
        self.assertEqual(result["localization"]["score_40"], 24)
        self.assertEqual(result["classification"]["score_30"], 15)
        self.assertEqual(result["total_score_70"], 39)

    def test_missing_prediction_scores_zero(self):
        result = self.score([], [self.truth(1)], 1)
        self.assertEqual(result["total_score_70"], 0)

    def test_duplicate_prediction_refuses_scoring(self):
        prediction = self.prediction(1)
        with self.assertRaises(ValidationError):
            self.score([prediction, prediction], [self.truth(1)], 1)


if __name__ == "__main__":
    unittest.main()
