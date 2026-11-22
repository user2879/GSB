from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slowdiag.canonical import canonical_bytes, canonical_json
from slowdiag.codecs import decode_record, encode_record
from slowdiag.config import default_config_dict, load_config_data
from slowdiag.errors import E_FORMAT, SlowDiagError
from slowdiag.generator import generate_all
from slowdiag.storage import read_all_sources


class CodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.config = load_config_data(default_config_dict(), self.workspace)
        generate_all(self.config)
        self.datasets = read_all_sources(self.config)
        self.app_source = self.config.source_by_id("app-prod")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_versions_are_decoded_and_padded_fields_are_explicit(self) -> None:
        evidence = [row for dataset in self.datasets for row in dataset.evidence]
        version_one = [row for row in evidence if row.record_version == 1]
        version_two = [row for row in evidence if row.record_version == 2]
        self.assertTrue(version_one)
        self.assertTrue(version_two)
        for row in version_one:
            self.assertIn("sample_period_ms", row.padded_fields)
        for row in version_two:
            self.assertNotIn("sample_period_ms", row.padded_fields)
        for row in evidence:
            self.assertGreaterEqual(row.elapsed_value, 1)
            self.assertGreaterEqual(row.scanned_value, 1)
            self.assertEqual(row.source_id, row.source_id.strip())

    def test_length_mismatch_is_e_format(self) -> None:
        payload = encode_record(
            "app_log_summary",
            2,
            {
                "query_id": "q001",
                "latency_p95_ms": 100,
                "latency_mean_ms": 80,
                "scanned_estimate": 1000,
                "scanned_sampled": 900,
                "sample_n": 2,
                "sample_period_ms": self.app_source.sample_period_ms,
            },
        )
        envelope = json.loads(payload)
        envelope["length"] = envelope["length"] + 1
        with self.assertRaises(SlowDiagError) as ctx:
            decode_record(self.app_source, 7, canonical_bytes(envelope))
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.source_id, "app-prod")
        self.assertEqual(ctx.exception.row_no, 7)
        self.assertEqual(ctx.exception.path, "$.length")

    def test_unknown_record_version_is_e_format(self) -> None:
        payload = canonical_bytes(
            {"length": 1, "version": 99, "body": {"x": 1}}
        )
        with self.assertRaises(SlowDiagError) as ctx:
            decode_record(self.app_source, 3, payload)
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.row_no, 3)
        self.assertEqual(ctx.exception.path, "$.version")

    def test_field_conflict_is_e_format(self) -> None:
        body = {
            "query_id": "q001",
            "latency_p95_ms": 100,
            "latency_mean_ms": 80,
            "scanned_estimate": 1000,
            "scanned_sampled": 900,
            "sample_n": 2,
            "sample_period_ms": self.app_source.sample_period_ms,
            "legacy_p95_ms": 101,
        }
        envelope = {
            "length": len(canonical_bytes(body)),
            "version": 2,
            "body": body,
        }
        with self.assertRaises(SlowDiagError) as ctx:
            decode_record(self.app_source, 4, canonical_bytes(envelope))
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.path, "$.body.legacy_p95_ms")

    def test_bool_numeric_value_is_e_format(self) -> None:
        body = {
            "query_id": "q001",
            "latency_p95_ms": True,
            "latency_mean_ms": 80,
            "scanned_estimate": 1000,
            "scanned_sampled": 900,
            "sample_n": 2,
            "sample_period_ms": self.app_source.sample_period_ms,
        }
        envelope = {"length": len(canonical_bytes(body)), "version": 2, "body": body}
        with self.assertRaises(SlowDiagError) as ctx:
            decode_record(self.app_source, 5, canonical_bytes(envelope))
        self.assertEqual(ctx.exception.code, E_FORMAT)
        self.assertEqual(ctx.exception.path, "$.body.latency_p95_ms")


if __name__ == "__main__":
    unittest.main()
