from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from edgecache.errors import SpecError
from edgecache.spec import load_manifest

from tests.helpers import ROOT


class SpecValidationTests(unittest.TestCase):
    def test_demo_manifest_loads(self) -> None:
        manifest = load_manifest(ROOT / "samples" / "demo.json")
        self.assertEqual([node.node_id for node in manifest.nodes], ["edge-a", "edge-b"])
        self.assertEqual(manifest.contents[0].shard_bytes, 4096)
        self.assertEqual([event.seq for event in manifest.events], list(range(1, 17)))

    def test_jsonl_reports_exact_first_invalid_line(self) -> None:
        with self.assertRaises(SpecError) as caught:
            load_manifest(ROOT / "samples" / "invalid" / "duplicate_port.jsonl")
        error = caught.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.line, 3)
        self.assertEqual(error.field, "port")
        self.assertIn("duplicates", error.message)

    def test_non_divisible_shards_are_rejected(self) -> None:
        data = {
            "format_version": 1,
            "config": {
                "sample_scale": 8,
                "lease_ttl_ms": 1000,
                "global_lease_limit": 1,
                "global_byte_budget": 1000,
            },
            "nodes": [
                {
                    "node_id": "n1",
                    "host": "127.0.0.1",
                    "port": 19001,
                    "slots": 1,
                    "lease_limit": 1,
                    "seed": 1,
                    "quota_version": 1,
                }
            ],
            "contents": [
                {"content_key": "c1", "total_bytes": 10, "shard_count": 3, "content_seed": 2}
            ],
            "events": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            text = json.dumps(data, indent=2)
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(SpecError) as caught:
                load_manifest(path)
        self.assertEqual(caught.exception.field, "shard_count")
        self.assertIn("divide", caught.exception.message)
        expected_line = next(
            index for index, line in enumerate(text.splitlines(), start=1) if '"shard_count"' in line
        )
        self.assertEqual(caught.exception.line, expected_line)

    def test_event_sequence_must_be_continuous(self) -> None:
        data = {
            "format_version": 1,
            "config": {
                "sample_scale": 8,
                "lease_ttl_ms": 1000,
                "global_lease_limit": 1,
                "global_byte_budget": 1000,
            },
            "nodes": [
                {
                    "node_id": "n1",
                    "host": "127.0.0.1",
                    "port": 19002,
                    "slots": 1,
                    "lease_limit": 1,
                    "seed": 1,
                    "quota_version": 1,
                }
            ],
            "contents": [
                {"content_key": "c1", "total_bytes": 4, "shard_count": 1, "content_seed": 2}
            ],
            "events": [
                {"seq": 1, "type": "apply", "node_id": "n1", "content_key": "c1", "shards": [0]},
                {"seq": 3, "type": "release", "lease_id": "L000001-S0000"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(SpecError) as caught:
                load_manifest(path)
        self.assertEqual(caught.exception.field, "seq")
        self.assertIn("contiguous", caught.exception.message)

    def test_host_must_be_literal_loopback(self) -> None:
        data = {
            "format_version": 1,
            "config": {
                "sample_scale": 8,
                "lease_ttl_ms": 1000,
                "global_lease_limit": 1,
                "global_byte_budget": 1000,
            },
            "nodes": [
                {
                    "node_id": "n1",
                    "host": "localhost",
                    "port": 19003,
                    "slots": 1,
                    "lease_limit": 1,
                    "seed": 1,
                    "quota_version": 1,
                }
            ],
            "contents": [
                {"content_key": "c1", "total_bytes": 4, "shard_count": 1, "content_seed": 2}
            ],
            "events": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(SpecError) as caught:
                load_manifest(path)
        self.assertEqual(caught.exception.field, "host")


if __name__ == "__main__":
    unittest.main()
