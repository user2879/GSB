from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rts_pipeline.sample import run_sample


class SampleIntegrationTests(unittest.TestCase):
    def test_complete_sample_corpus_and_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            report = run_sample(Path(root) / "samples")
            self.assertTrue(report["all_checks_passed"])
            self.assertTrue(
                report["reproducibility"]["byte_identical_scenario_trees"]
            )
            self.assertEqual(
                report["reproducibility"]["primary_tree_hash"],
                report["reproducibility"]["reproduction_tree_hash"],
            )
            self.assertEqual(
                report["scenarios"]["overflow"]["result_code"],
                "E_RESOURCE_OVERFLOW",
            )
            self.assertLessEqual(
                report["scenarios"]["budget_tight"]["effective_parallelism"],
                1,
            )


if __name__ == "__main__":
    unittest.main()

