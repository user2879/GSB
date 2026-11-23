from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_samples import generate


class GeneratorTests(unittest.TestCase):
    def test_fixed_seed_generation_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as left_temp, tempfile.TemporaryDirectory() as right_temp:
            left = Path(left_temp)
            right = Path(right_temp)
            left_manifest = generate(left)
            right_manifest = generate(right)
            self.assertEqual(left_manifest, right_manifest)
            for name in (
                "exposure_samples.jsonl",
                "feature_config.json",
                "train_config.yaml",
                "commands.json",
                "manifest.json",
            ):
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
