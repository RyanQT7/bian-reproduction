import unittest

from bian.predictions import validate_predictions


class PredictionPartialFailureTests(unittest.TestCase):
    def test_classification_failure_preserves_localization(self):
        record = {
            "incident_id": "i1",
            "prediction_status": "success",
            "classification_status": "classification_failed",
            "classification_error": "model format failure",
            "top5_root_causes": [
                {"rank": i, "node_id": f"n{i}", "failure_score": 0.2}
                for i in range(1, 6)
            ],
            "rank_of_ranks": {
                "rounds": 2,
                "raw_rankings": [[f"n{i}" for i in range(1, 6)]] * 2,
            },
        }
        result = validate_predictions(
            [record],
            expected_incident_ids={"i1"},
            candidate_node_ids={f"n{i}" for i in range(1, 6)},
            taxonomy=[],
            expected_rank_rounds=2,
        )
        self.assertTrue(result["valid"], result["errors"])


if __name__ == "__main__":
    unittest.main()
