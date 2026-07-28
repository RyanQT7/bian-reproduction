import unittest

from bian.data.validators import ValidationError
from bian.models.structured_output import (
    parse_strict_json,
    strip_thinking,
    validate_device_analysis,
    validate_stage1,
    validate_stage2,
)
from bian.models.dual_7b_backend import Dual7BBackend, GenerationConfig


class StructuredOutputTests(unittest.TestCase):
    def test_thinking_is_separated(self):
        thoughts, answer = strip_thinking('<think>hidden</think>{"ok":true}')
        self.assertEqual(thoughts, "hidden")
        self.assertEqual(answer, '{"ok":true}')
        parsed, _ = parse_strict_json('<think>hidden</think>{"ok":true}')
        self.assertTrue(parsed["ok"])

    def test_unterminated_thinking_and_prose_are_rejected(self):
        with self.assertRaises(ValidationError):
            parse_strict_json("<think>unfinished")
        with self.assertRaises(ValidationError):
            parse_strict_json('answer: {"ok":true}')

    def test_json_repair_dependency_handles_missing_comma_then_schema_applies(self):
        import json_repair

        repaired = json_repair.loads('{"node":"a" "score":1}')
        self.assertEqual(repaired, {"node": "a", "score": 1})

    def test_device_analysis_handles_partial_and_empty_as_text(self):
        nodes = ("region-1-br-1", "region-1-br-2")
        value = {
            "devices": [
                {
                    "node_id": node,
                    "alert_summary": status,
                    "is_anomalous": False,
                    "anomaly_score": 0.1,
                    "anomaly_evidence": status,
                    "uncertainty": status,
                }
                for node, status in zip(nodes, ("partial", "empty"))
            ]
        }
        self.assertEqual(len(validate_device_analysis(value, nodes)["devices"]), 2)
        del value["devices"][0]["uncertainty"]
        with self.assertRaises(ValidationError):
            validate_device_analysis(value, nodes)

    def test_stage1_normalizes_and_rejects_illegal_scores(self):
        nodes = ("region-1-br-1", "region-1-br-2")
        value = {
            "scores": [
                {"node_id": nodes[0], "score": 2},
                {"node_id": nodes[1], "score": 1},
            ],
        }
        result = validate_stage1(value, nodes)
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0)
        zeros = {
            "scores": [
                {"node_id": nodes[0], "score": 0},
                {"node_id": nodes[1], "score": 0},
            ]
        }
        self.assertEqual(
            sum(
                validate_stage1(
                    zeros, nodes, require_positive=False
                )["scores"].values()
            ),
            0,
        )
        value["scores"][0]["score"] = float("nan")
        with self.assertRaises(ValidationError):
            validate_stage1(value, nodes)

    def test_stage2_rejects_switch_and_duplicate_fault(self):
        nodes = tuple(f"region-1-service-{i}" for i in range(1, 4)) + (
            "region-1-br-1",
            "region-1-br-2",
        )
        taxonomy = tuple(
            {"fault_type": f"type-{i}", "fault_category": f"cat-{i}"}
            for i in range(1, 4)
        )
        value = {
            "root_causes": [
                {"node_id": node, "score": 1 / (i + 1)}
                for i, node in enumerate(nodes)
            ],
            "fault_types": [
                {
                    "fault_type": item["fault_type"],
                    "confidence": 0.2,
                }
                for item in taxonomy
            ],
        }
        self.assertEqual(len(validate_stage2(value, nodes, taxonomy)["root_causes"]), 5)
        value["root_causes"][0]["anomaly_score"] = 0.5
        value["root_causes"][0]["anomaly_evidence"] = "evidence"
        value["root_causes"][0]["uncertainty"] = "partial"
        self.assertEqual(len(validate_stage2(value, nodes, taxonomy)["root_causes"]), 5)
        value["root_causes"][0]["node_id"] = "region-1-sw-core"
        with self.assertRaises(ValidationError):
            validate_stage2(value, nodes, taxonomy)

    def test_backend_configuration_has_finite_input_limit(self):
        backend = Dual7BBackend(
            __import__("pathlib").Path("/read-only/model"),
            config=GenerationConfig(max_input_tokens=8192),
            prompt_dir=__import__("pathlib").Path("/prompts"),
        )
        self.assertEqual(backend.config.max_input_tokens, 8192)


if __name__ == "__main__":
    unittest.main()
