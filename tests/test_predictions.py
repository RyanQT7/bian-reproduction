import json
from pathlib import Path
import stat
import tempfile
import unittest

from bian.data.validators import ValidationError
from bian.predictions import freeze_predictions, validate_predictions


class PredictionValidationTests(unittest.TestCase):
    candidates = {f"region-1-service-{index}" for index in range(1, 4)} | {
        "region-1-br-1",
        "region-1-br-2",
    }
    taxonomy = [
        {"fault_type": "type-a", "fault_category": "cat-a"},
        {"fault_type": "type-b", "fault_category": "cat-b"},
        {"fault_type": "type-c", "fault_category": "cat-c"},
    ]

    def record(self):
        nodes = sorted(self.candidates)
        return {
            "incident_id": "incident-0001",
            "top5_root_causes": [
                {
                    "rank": index + 1,
                    "node_id": node,
                    "failure_score": 0.2,
                    "reason_summary": "test",
                }
                for index, node in enumerate(nodes)
            ],
            "predicted_fault_type": "type-a",
            "predicted_fault_category": "cat-a",
            "fault_type_top3": [
                {
                    "rank": index + 1,
                    "fault_type": fault_type,
                    "fault_category": f"cat-{fault_type[-1]}",
                    "confidence": 1 / 3,
                    "reason_summary": "test",
                }
                for index, fault_type in enumerate(("type-a", "type-b", "type-c"))
            ],
            "rank_of_ranks": {
                "rounds": 3,
                "raw_rankings": [nodes, nodes, nodes],
            },
        }

    def validate(self, records):
        return validate_predictions(
            records,
            expected_incident_ids={"incident-0001"},
            candidate_node_ids=self.candidates,
            taxonomy=self.taxonomy,
            expected_rank_rounds=3,
        )

    def test_valid_prediction(self):
        self.assertTrue(self.validate([self.record()])["valid"])

    def test_duplicate_incident_rejected(self):
        result = self.validate([self.record(), self.record()])
        self.assertFalse(result["valid"])
        self.assertTrue(any("duplicate" in error for error in result["errors"]))

    def test_illegal_node_and_score_rejected(self):
        record = self.record()
        record["top5_root_causes"][0]["node_id"] = "region-1-sw-core"
        record["top5_root_causes"][0]["failure_score"] = float("inf")
        result = self.validate([record])
        self.assertFalse(result["valid"])

    def test_freeze_is_read_only_and_no_overwrite(self):
        record = self.record()
        validation = self.validate([record])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.jsonl"
            source.write_text(json.dumps(record) + "\n")
            result = freeze_predictions(
                predictions_path=source,
                output_dir=root,
                validation=validation,
                manifest={"test": True},
            )
            frozen = Path(result["frozen_path"])
            self.assertFalse(frozen.stat().st_mode & stat.S_IWUSR)
            with self.assertRaises(FileExistsError):
                freeze_predictions(
                    predictions_path=source,
                    output_dir=root,
                    validation=validation,
                    manifest={"test": True},
                )

    def test_explicit_failed_prediction_is_valid_but_cannot_contain_answers(self):
        failed = {
            "incident_id": "incident-0001",
            "prediction_status": "prediction_failed",
            "error_type": "ValidationError",
            "error": "model output invalid after retries",
        }
        self.assertTrue(self.validate([failed])["valid"])
        failed["top5_root_causes"] = self.record()["top5_root_causes"]
        self.assertFalse(self.validate([failed])["valid"])


if __name__ == "__main__":
    unittest.main()
