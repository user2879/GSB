import copy
import json
import os
import tempfile
import unittest

from shardflow.errors import SpecError
from shardflow.spec import load_spec, topological_order

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "demo", "demo.spec.json")
EVENTS_PATH = os.path.join(ROOT, "demo", "demo.events.ndjson")


class SpecValidationTests(unittest.TestCase):
    def setUp(self):
        with open(SPEC_PATH, encoding="utf-8") as handle:
            self.doc = json.load(handle)
        with open(EVENTS_PATH, encoding="utf-8") as handle:
            self.events = [json.loads(line) for line in handle if line.strip()]

    def _write(self, tmp, doc, events=None):
        spec_path = os.path.join(tmp, "spec.json")
        events_path = os.path.join(tmp, "events.ndjson")
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        with open(events_path, "w", encoding="utf-8") as handle:
            for event in events or self.events:
                handle.write(json.dumps(event) + "\n")
        return spec_path, events_path

    def test_valid_demo_loads_with_stable_topo_order(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        self.assertEqual(spec.topo_order,
                         ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"])

    def test_duplicate_node_id_reports_object_and_field(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"][1]["node_id"] = doc["nodes"][0]["node_id"]
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, events_path = self._write(tmp, doc)
            with self.assertRaises(SpecError) as ctx:
                load_spec(spec_path, events_path)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "node_id")

    def test_non_loopback_host_rejected(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"][0]["host"] = "10.0.0.1"
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, events_path = self._write(tmp, doc)
            with self.assertRaises(SpecError) as ctx:
                load_spec(spec_path, events_path)
        self.assertEqual(ctx.exception.field, "host")

    def test_duplicate_port_rejected(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"][1]["port"] = doc["nodes"][0]["port"]
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, events_path = self._write(tmp, doc)
            with self.assertRaises(SpecError) as ctx:
                load_spec(spec_path, events_path)
        self.assertEqual(ctx.exception.field, "port")

    def test_dependency_cycle_rejected(self):
        doc = copy.deepcopy(self.doc)
        doc["shards"][0]["depends_on"] = ["s3"]
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, events_path = self._write(tmp, doc)
            with self.assertRaises(SpecError) as ctx:
                load_spec(spec_path, events_path)
        self.assertEqual(ctx.exception.field, "depends_on")

    def test_revision_parent_must_exist(self):
        doc = copy.deepcopy(self.doc)
        doc["revisions"][1]["parent"] = 99
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, events_path = self._write(tmp, doc)
            with self.assertRaises(SpecError) as ctx:
                load_spec(spec_path, events_path)
        self.assertEqual(ctx.exception.field, "parent")

    def test_cancel_without_dispatch_rejected(self):
        events = [{"type": "cancel", "shard_id": "s1"}]
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, events_path = self._write(tmp, self.doc, events)
            with self.assertRaises(SpecError) as ctx:
                load_spec(spec_path, events_path)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_positive_integer_enforcement(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"][0]["concurrency"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            spec_path, events_path = self._write(tmp, doc)
            with self.assertRaises(SpecError) as ctx:
                load_spec(spec_path, events_path)
        self.assertEqual(ctx.exception.field, "concurrency")

    def test_topological_order_is_deterministic(self):
        spec = load_spec(SPEC_PATH, EVENTS_PATH)
        shuffled = dict(sorted(spec.shards.items(), reverse=True))
        self.assertEqual(topological_order(shuffled), spec.topo_order)


if __name__ == "__main__":
    unittest.main()
