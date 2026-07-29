import unittest

from bian.methods.evidence_budget import select_evidence
from bian.models.vllm_schemas import classification_schema, stage2_schema


class VLLMSchemaTests(unittest.TestCase):
    def test_stage2_schema_constrains_aliases_and_evidence(self):
        schema = stage2_schema({"C01", "C02"}, {"E01", "E02"})
        item = schema["properties"]["candidates"]["items"]["properties"]
        self.assertEqual(item["candidate_id"]["enum"], ["C01", "C02"])
        self.assertEqual(
            item["supporting_evidence_ids"]["items"]["enum"], ["E01", "E02"]
        )
        self.assertFalse(schema["additionalProperties"])

    def test_type_schema_uses_const_type_id(self):
        schema = classification_schema("T07", {"C01"}, {"E01"})
        self.assertEqual(schema["properties"]["type_id"]["const"], "T07")
        aliases = schema["properties"]["root_hypotheses"]["items"]["properties"]
        self.assertEqual(aliases["candidate_id"]["enum"], ["C01"])

    def test_budget_prioritizes_referenced_and_direct_evidence(self):
        stage1 = {
            "supporting_evidence_ids": ["support"],
            "counter_evidence_ids": ["counter"],
        }
        evidence = {
            "evidence": [
                {
                    "evidence_id": "large",
                    "stable_change_score": 0.99,
                    "direct_fault_evidence": False,
                    "data_quality_status": "available",
                },
                {
                    "evidence_id": "direct",
                    "stable_change_score": 0.2,
                    "direct_fault_evidence": True,
                    "data_quality_status": "available",
                },
                {
                    "evidence_id": "counter",
                    "stable_change_score": 0,
                    "direct_fault_evidence": False,
                    "data_quality_status": "available",
                },
                {
                    "evidence_id": "support",
                    "stable_change_score": 0,
                    "direct_fault_evidence": False,
                    "data_quality_status": "available",
                },
            ]
        }
        selected = select_evidence(stage1, evidence, max_evidence=3)
        self.assertEqual(
            [item["evidence_id"] for item in selected],
            ["support", "counter", "direct"],
        )

    def test_unavailable_is_not_promoted_as_direct(self):
        stage1 = {}
        evidence = {
            "evidence": [
                {
                    "evidence_id": "unavailable",
                    "stable_change_score": 1,
                    "direct_fault_evidence": True,
                    "data_quality_status": "unavailable_by_role",
                },
                {
                    "evidence_id": "changed",
                    "stable_change_score": 0.1,
                    "direct_fault_evidence": False,
                    "first_change_time": "2026-01-01T00:00:00Z",
                    "data_quality_status": "available",
                },
            ]
        }
        selected = select_evidence(stage1, evidence, max_evidence=1)
        self.assertEqual(selected[0]["evidence_id"], "changed")


if __name__ == "__main__":
    unittest.main()
