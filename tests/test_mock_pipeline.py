import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from bian import VALIDATION_LABEL
from bian.data.mock_generator import generate_mock_incidents
from bian.data.validators import validate_scores
from bian.models.mock_backend import MockModelBackend
from bian.pipeline import run_incident


class MockPipelineTests(unittest.TestCase):
    def run_once(self, seed=17):
        incident = generate_mock_incidents(seed, 1, 4)[0]
        result = run_incident(
            incident,
            MockModelBackend(seed),
            rank_rounds=3,
            entropy_threshold=0.75,
            entropy_mode="raw",
            stage2_device_top_p=0.8,
        )
        return incident, result

    def test_seed_is_deterministic(self):
        self.assertEqual(self.run_once()[1], self.run_once()[1])

    def test_pipeline_contracts(self):
        incident, result = self.run_once()
        self.assertEqual(result["result_label"], VALIDATION_LABEL)
        validate_scores(result["stage1_scores"], incident.candidate_devices)
        self.assertTrue(set(result["ranking"]) <= set(incident.candidate_devices))
        self.assertTrue(set(result["retained_candidates"]) <= set(result["subgraph"]["nodes"]))
        timestamps = [item["timestamp"] for item in result["timeline"]]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(timestamps), len(set(timestamps)))

    def test_cli_creates_required_artifacts(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            output = Path(temporary) / "run"
            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "scripts.run_mock",
                    "--config",
                    "configs/experiments/mock_pipeline.yaml",
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            required = {
                "summary.md",
                "metrics.json",
                "predictions.jsonl",
                "resolved_config.yaml",
                "environment.txt",
                "run.log",
                "tests.log",
                "changed_files.txt",
            }
            self.assertEqual(required, {item.name for item in output.iterdir()})
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["result_label"], VALIDATION_LABEL)


if __name__ == "__main__":
    unittest.main()
