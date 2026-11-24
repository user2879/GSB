from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quota_ledger.errors import BusinessError
from quota_ledger.spec import load_spec

from .helpers import make_sample


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample, self.base = make_sample(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _mutate(self, filename: str, fn) -> None:
        path = self.sample / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        fn(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_valid_sample(self) -> None:
        spec = load_spec(self.sample)
        self.assertEqual(len(spec.nodes), 3)
        self.assertEqual(len(spec.shards), 10)
        self.assertEqual(spec.topo_order, sorted(spec.topo_order))
        self.assertEqual(spec.revisions[2].parent, 1)

    def test_duplicate_node_id_has_position_and_field(self) -> None:
        def mutate(doc):
            doc["nodes"][1]["id"] = doc["nodes"][0]["id"]
        self._mutate("nodes.json", mutate)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "id")
        self.assertIn("nodes[1]", ctx.exception.position)

    def test_duplicate_port(self) -> None:
        def mutate(doc):
            doc["nodes"][1]["port"] = doc["nodes"][0]["port"]
        self._mutate("nodes.json", mutate)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("E_SPEC", "port"))

    def test_non_literal_loopback_host(self) -> None:
        def mutate(doc):
            doc["nodes"][0]["host"] = "localhost"
        self._mutate("nodes.json", mutate)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("E_SPEC", "host"))

    def test_dependency_cycle(self) -> None:
        def mutate(doc):
            doc["shards"][0]["dependencies"].append(doc["shards"][-1]["id"])
            doc["shards"][-1]["dependencies"].append(doc["shards"][0]["id"])
        self._mutate("shards.json", mutate)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual(ctx.exception.code, "E_GRAPH")
        self.assertEqual(ctx.exception.field, "dependencies")
        self.assertIn("cycle", ctx.exception.detail or {})

    def test_missing_dependency(self) -> None:
        def mutate(doc):
            doc["shards"][0]["dependencies"] = ["shard-does-not-exist"]
        self._mutate("shards.json", mutate)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("E_GRAPH", "dependencies"))

    def test_revision_parent_must_exist(self) -> None:
        def mutate(doc):
            doc["revisions"][1]["parent"] = 99
        self._mutate("revisions.json", mutate)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("E_SPEC", "parent"))

    def test_event_sequence_must_increase(self) -> None:
        def mutate(doc):
            doc["events"][2]["seq"] = doc["events"][1]["seq"]
        self._mutate("events.json", mutate)
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual((ctx.exception.code, ctx.exception.field), ("E_SPEC", "seq"))

    def test_malformed_json_position(self) -> None:
        (self.sample / "nodes.json").write_text('{"nodes":[', encoding="utf-8")
        with self.assertRaises(BusinessError) as ctx:
            load_spec(self.sample)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("nodes.json:", ctx.exception.position or "")
        self.assertEqual(ctx.exception.field, "$")


if __name__ == "__main__":
    unittest.main()
