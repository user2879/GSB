from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cm_fusion.compat import CompatTable
from cm_fusion.config import Config
from cm_fusion.loaders import Document, Query
from cm_fusion.postings import build_logical_index
from cm_fusion.sample_data import generate_samples
from cm_fusion.search import execute_search


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        samples = Path(cls.temp.name) / "samples"
        generate_samples(samples, 20261120)
        cls.config = Config.load(samples / "final" / "config.json", CompatTable.load(samples / "final" / "compat.json"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_title_only_document_is_recalled(self) -> None:
        document = Document("D-title-only", ("Alpha", "Beta"), ("Zeta", "Eta"), "shard-a", 1)
        query = Query("Q-title-only", ("Alpha", "Gamma"), "q1", 1)
        logical = build_logical_index((document,), self.config)
        outcome = execute_search((document,), (query,), logical, self.config)
        result = outcome.results[0]
        self.assertEqual(result["candidate_count"], 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["doc_id"], "D-title-only")
        self.assertEqual(candidate["lexical"]["overlap_count"], 1)
        self.assertEqual(candidate["cooccurrence"]["dot"], 0)
        self.assertGreater(candidate["lexical"]["contribution_fixed"], 0)


if __name__ == "__main__":
    unittest.main()

