from __future__ import annotations

import unittest

from spectral_diag.config import RunConfig
from spectral_diag.pipeline import execute_pipeline
from spectral_diag.scene import generate_scene


class DeterminismTests(unittest.TestCase):
    def test_scene_is_byte_identical_for_same_seed(self) -> None:
        first = generate_scene(RunConfig())
        second = generate_scene(RunConfig())
        self.assertEqual(first.canonical_bytes, second.canonical_bytes)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.payload["reference_truth"]["density_nm"], 0.5)
        self.assertEqual(first.payload["reference_truth"]["observer"], "cie1931_2")

    def test_pipeline_result_hash_is_stable(self) -> None:
        first = execute_pipeline(RunConfig())
        second = execute_pipeline(RunConfig())
        self.assertEqual(first["status"], "accepted")
        self.assertEqual(first["result_sha256"], second["result_sha256"])
        self.assertEqual(first["integration"]["normalization"], "y100_reflector")
        self.assertEqual(first["integration"]["density_nm"], 0.5)

    def test_seed_changes_scene(self) -> None:
        first = generate_scene(RunConfig(seed=1))
        second = generate_scene(RunConfig(seed=2))
        self.assertNotEqual(first.sha256, second.sha256)


if __name__ == "__main__":
    unittest.main()
