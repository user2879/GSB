from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cm_fusion.compat import CompatTable
from cm_fusion.config import Config
from cm_fusion.errors import ValidationError
from cm_fusion.loaders import load_documents, load_queries
from cm_fusion.normalization import NormalizationRules, normalize_term
from cm_fusion.sample_data import generate_samples
from cm_fusion.util import deep_merge
from cm_fusion.varint import decode_uvarint, encode_uvarint


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.samples = Path(cls.temp.name) / "samples"
        generate_samples(cls.samples, 20261120)
        cls.compat = CompatTable.load(cls.samples / "compat.json")
        cls.config = Config.load(cls.samples / "config.json", cls.compat)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_uvarint_roundtrip(self) -> None:
        for value in (0, 1, 127, 128, 16384, 2**32, 2**63 - 1):
            encoded = encode_uvarint(value)
            decoded, offset = decode_uvarint(encoded)
            self.assertEqual(decoded, value)
            self.assertEqual(offset, len(encoded))

    def test_normalization_folds_case_width_and_whitespace(self) -> None:
        rules = NormalizationRules(True, True, True)
        self.assertEqual(normalize_term("  ＡＬＰＨＡ\t ", rules), "alpha")
        self.assertEqual(normalize_term("Fusion", rules), "fusion")
        self.assertEqual(normalize_term("检索", rules), "检索")

    def test_config_validation_rejects_bad_block_and_tier(self) -> None:
        with self.assertRaises(ValidationError) as block_error:
            Config.from_mapping(deep_merge(self.config.raw, {"posting_block_size": 0}), self.compat)
        self.assertEqual(block_error.exception.code, "E_INDEX_INVALID")
        with self.assertRaises(ValidationError) as tier_error:
            Config.from_mapping(deep_merge(self.config.raw, {"active_compression_tier": 3}), self.compat)
        self.assertEqual(tier_error.exception.code, "E_INDEX_INVALID")
        with self.assertRaises(ValidationError) as bits_error:
            Config.from_mapping(deep_merge(self.config.raw, {"fixed_decimal_bits": 31}), self.compat)
        self.assertEqual(bits_error.exception.code, "E_INDEX_INVALID")

    def test_loader_error_codes_and_failure_order(self) -> None:
        documents, doc_failures, _ = load_documents(self.samples / "invalid_docs.jsonl", self.config)
        queries, query_failures, _ = load_queries(self.samples / "invalid_queries.jsonl", self.config)
        failures = doc_failures + query_failures
        self.assertEqual([doc.doc_id for doc in documents], ["D8005"])
        self.assertEqual([query.query_id for query in queries], ["Q8004"])
        self.assertEqual(len(failures), 9)
        self.assertEqual(
            {failure.code for failure in failures},
            {"E_DOC_INVALID", "E_INDEX_INVALID"},
        )
        ordered = sorted(failures, key=lambda item: item.sort_key())
        self.assertEqual(ordered, sorted(ordered, key=lambda item: item.sort_key()))


if __name__ == "__main__":
    unittest.main()

