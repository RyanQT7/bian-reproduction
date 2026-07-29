import unittest
from pathlib import Path

from bian.models.dual_7b_backend import GenerationConfig
from bian.models.sharded_backend import Sharded32BBackend


class ShardedBackendTests(unittest.TestCase):
    def test_configuration_is_offline_and_explicit(self):
        backend = Sharded32BBackend(
            Path("/read-only/model"),
            config=GenerationConfig(max_input_tokens=16384),
            prompt_dir=Path("/prompts"),
            device_map="balanced",
            max_memory={0: "44GiB", 1: "44GiB"},
        )
        self.assertEqual(backend.precision, "bfloat16")
        self.assertEqual(backend.max_memory[0], "44GiB")
        self.assertEqual(backend.config.max_input_tokens, 16384)


if __name__ == "__main__":
    unittest.main()
