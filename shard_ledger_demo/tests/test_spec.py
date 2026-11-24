import json
import os
import tempfile
import unittest

from shard_ledger.errors import CycleError, DependError, SpecError
from shard_ledger.spec import load_events, load_manifest

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "manifest.json")
EVENTS = os.path.join(os.path.dirname(__file__), "..", "samples", "events.ndjson")


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        with open(SAMPLE, encoding="utf-8") as handle:
            self.data = json.load(handle)

    def _load(self, data):
        path = os.path.join(self.tmp, "m.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        return load_manifest(path)

    def test_sample_manifest_and_events_are_valid(self):
        manifest = load_manifest(SAMPLE)
        events = load_events(EVENTS, manifest)
        self.assertEqual(len(manifest.nodes), 3)
        self.assertEqual(len(events), 10)
        self.assertEqual(manifest.order[0], "S1")

    def test_duplicate_node_id_rejected(self):
        self.data["nodes"][1]["node_id"] = self.data["nodes"][0]["node_id"]
        with self.assertRaises(SpecError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "node_id")

    def test_non_loopback_host_rejected(self):
        self.data["nodes"][0]["host"] = "10.0.0.1"
        with self.assertRaises(SpecError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.field, "host")

    def test_duplicate_port_rejected(self):
        self.data["nodes"][1]["port"] = self.data["nodes"][0]["port"]
        with self.assertRaises(SpecError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.field, "port")

    def test_non_positive_quota_rejected(self):
        self.data["nodes"][0]["concurrency"] = 0
        with self.assertRaises(SpecError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.field, "concurrency")

    def test_missing_dependency_reference(self):
        self.data["shards"][0]["deps"] = ["GHOST"]
        with self.assertRaises(DependError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.code, "E_DEPEND")
        self.assertEqual(ctx.exception.field, "deps")

    def test_unknown_target_node(self):
        self.data["shards"][0]["targets"] = ["N9"]
        with self.assertRaises(DependError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.field, "targets")

    def test_cycle_rejected_with_first_entity(self):
        self.data["shards"][0]["deps"] = ["S3"]
        with self.assertRaises(CycleError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.code, "E_CYCLE")
        self.assertIn(ctx.exception.entity, {"S1", "S3"})

    def test_revision_must_be_monotonic(self):
        self.data["revisions"][1]["version"] = 1
        with self.assertRaises(SpecError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.field, "version")

    def test_revision_parent_must_reference_previous(self):
        self.data["revisions"][1]["parent"] = 9
        with self.assertRaises(SpecError) as ctx:
            self._load(self.data)
        self.assertEqual(ctx.exception.field, "parent")

    def test_corrupt_json_rejected(self):
        path = os.path.join(self.tmp, "broken.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{ not json")
        with self.assertRaises(SpecError) as ctx:
            load_manifest(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_event_seq_must_be_consecutive(self):
        manifest = load_manifest(SAMPLE)
        path = os.path.join(self.tmp, "e.ndjson")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"seq": 5, "type": "submit", "shard_id": "S1"}) + "\n")
        with self.assertRaises(SpecError) as ctx:
            load_events(path, manifest)
        self.assertEqual(ctx.exception.field, "seq")


if __name__ == "__main__":
    unittest.main()
