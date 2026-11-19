from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cm_fusion.compat import CompatTable
from cm_fusion.config import Config
from cm_fusion.loaders import load_corpus
from cm_fusion.sample_data import generate_samples
from cm_fusion.verification import (
    verify_conversion,
    verify_dual_tier,
    verify_position_switch,
    verify_validation,
)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.samples = Path(cls.temp.name) / "samples"
        generate_samples(cls.samples, 20261120)
        cls.compat = CompatTable.load(cls.samples / "final" / "compat.json")
        cls.config = Config.load(cls.samples / "final" / "config.json", cls.compat)
        cls.corpus = load_corpus(
            cls.samples / "final" / "docs.jsonl",
            cls.samples / "final" / "queries.jsonl",
            cls.config,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_dual_tier_lossless(self) -> None:
        result = verify_dual_tier(self.config, self.corpus.documents)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tier1"]["digest"], result["tier2"]["digest"])
        self.assertNotEqual(result["tier1"]["fingerprint"], result["tier2"]["fingerprint"])
        self.assertNotEqual(result["tier1"]["semantic"], result["tier2"]["semantic"])

    def test_position_switch_is_structural(self) -> None:
        result = verify_position_switch(self.config, self.corpus.documents)
        self.assertTrue(result["ok"])
        self.assertTrue(result["with_positions_decoded"])
        self.assertTrue(result["without_positions_decoded"])
        self.assertTrue(result["term_frequencies_equal"])
        self.assertTrue(result["structural_flag_distinct"])

    def test_v1_v2_roundtrip_and_sidecar_losses(self) -> None:
        result = verify_conversion(self.config, self.corpus.documents)
        self.assertTrue(result["ok"])
        self.assertEqual(result["v2_digest"], result["v1_digest"])
        self.assertEqual(result["v1_digest"], result["v2_back_digest"])
        self.assertIn("position_skiplist", result["v1_loss_features"])
        self.assertIn("block_checksum", result["v2_back_loss_features"])
        self.assertNotEqual(result["tier_mapping"]["source_v2"], result["tier_mapping"]["converted_v1"])

    def test_validation_report(self) -> None:
        result = verify_validation(self.samples, self.config)
        self.assertTrue(result["ok"])
        self.assertEqual(result["failure_count"], 9)
        self.assertIn("E_DOC_INVALID", result["codes"])
        self.assertIn("E_INDEX_INVALID", result["codes"])


if __name__ == "__main__":
    unittest.main()

