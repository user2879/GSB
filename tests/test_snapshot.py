import json
import tempfile
import unittest
from pathlib import Path

from cache_invalidation.engine import Engine
from cache_invalidation.errors import EFormat, ESnapshot
from cache_invalidation.model import EntryID
from cache_invalidation.snapshot import (
    DEFAULT_NAMESPACE,
    decode_snapshot_text,
    encode_snapshot,
)
from cache_invalidation.store import Store
from tests.support import sample_graph


class SnapshotCodecTest(unittest.TestCase):
    def setUp(self):
        self.graph = sample_graph()
        engine = Engine(self.graph, store=Store(), format_version=3)
        result = engine.explicit_invalidate(EntryID("catalog", 100), 100)
        self.rounds = engine.records
        self.versions = dict(engine.versions)

    def test_v3_round_trip_preserves_graph_and_rounds(self):
        text = encode_snapshot(self.graph, self.rounds, 3)
        data = decode_snapshot_text(text)
        self.assertEqual(data.format_version, 3)
        self.assertEqual(
            data.graph.namespace_parents["storefront"],
            ("pricing", "inventory"))
        self.assertEqual(
            data.graph.entries[EntryID("catalog", 100)].digest_algorithm,
            "crc32")
        self.assertEqual(len(data.rounds), 1)
        self.assertEqual(
            list(data.rounds[0].affected),
            list(self.rounds[0].affected))

    def test_v2_contains_annotations_but_no_namespaces(self):
        text = encode_snapshot(self.graph, self.rounds, 2)
        payload = json.loads(text)
        self.assertNotIn("namespaces", payload)
        entry = next(item for item in payload["entries"]
                     if item["key"] == 100)
        self.assertIn("annotations", entry)
        self.assertNotIn("digest_algorithm", entry)
        data = decode_snapshot_text(text)
        self.assertEqual(
            sorted(entry_id.namespace for entry_id in data.graph.entries),
            [DEFAULT_NAMESPACE] * 6)
        targets = data.graph.annotation_edges[
            EntryID(DEFAULT_NAMESPACE, 100)]
        self.assertEqual(
            sorted(target.key for target in targets), [200, 300])

    def test_v1_only_contains_key_and_expiry(self):
        text = encode_snapshot(self.graph, self.rounds, 1)
        payload = json.loads(text)
        self.assertEqual(
            sorted(payload["entries"][0].keys()),
            ["expire_ms", "key"])
        data = decode_snapshot_text(text)
        self.assertEqual(len(data.graph.entries), 6)
        self.assertEqual(
            data.rounds[0].affected[0],
            f"{DEFAULT_NAMESPACE}/100")

    def test_unknown_format_version_is_e_format(self):
        payload = json.loads(
            encode_snapshot(self.graph, self.rounds, 3))
        payload["format_version"] = 99
        with self.assertRaises(EFormat) as caught:
            decode_snapshot_text(json.dumps(payload))
        self.assertEqual(caught.exception.field_path, "format_version")

    def test_missing_required_field_reports_path_and_line(self):
        text = "{\n  \"format_version\": 3,\n  \"state_version\": 1\n}"
        with self.assertRaises(EFormat) as caught:
            decode_snapshot_text(text)
        self.assertEqual(caught.exception.field_path, "entries")
        self.assertEqual(caught.exception.line, 1)

    def test_unsupported_digest_algorithm_is_e_format(self):
        payload = json.loads(
            encode_snapshot(self.graph, self.rounds, 3))
        payload["entries"][0]["digest_algorithm"] = "md5"
        with self.assertRaises(EFormat) as caught:
            decode_snapshot_text(json.dumps(payload))
        self.assertEqual(
            caught.exception.field_path,
            "entries[0].digest_algorithm")

    def test_truncated_snapshot_is_e_snapshot_with_line(self):
        text = encode_snapshot(self.graph, self.rounds, 3)
        with self.assertRaises(ESnapshot) as caught:
            decode_snapshot_text(text[: int(len(text) * 0.6)])
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        self.assertIsNotNone(caught.exception.line)
        self.assertTrue(caught.exception.details["incomplete"])

    def test_snapshot_bytes_are_deterministic(self):
        first = encode_snapshot(self.graph, self.rounds, 3)
        second = encode_snapshot(self.graph, self.rounds, 3)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
