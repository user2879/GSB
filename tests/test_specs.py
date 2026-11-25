from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lease_arbitrator.errors import SpecError
from lease_arbitrator.sample import demo_documents, write_sample
from lease_arbitrator.specs import load_bundle


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name) / "sample"
        write_sample(self.dir)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def mutate_nodes(self, mutate) -> None:
        path = self.dir / "nodes.json"
        doc = json.loads(path.read_text())
        mutate(doc)
        path.write_text(json.dumps(doc))

    def mutate_pipeline(self, mutate) -> None:
        path = self.dir / "pipeline.json"
        doc = json.loads(path.read_text())
        mutate(doc)
        path.write_text(json.dumps(doc))

    def rewrite_events(self, events) -> None:
        (self.dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_valid_sample_loads(self) -> None:
        bundle = load_bundle(self.dir)
        self.assertEqual(len(bundle.nodes), 3)
        self.assertEqual(bundle.events[-1].seq, 12)

    def test_node_id_port_and_seed_validation(self) -> None:
        cases = [
            (lambda doc: doc["nodes"][1].__setitem__("node_id", "node-a"), "node-a", "node_id"),
            (lambda doc: doc["nodes"][0].__setitem__("host", "0.0.0.0"), "node-a", "host"),
            (lambda doc: doc["nodes"][1].__setitem__("port", doc["nodes"][0]["port"]), "node-b", "port"),
            (lambda doc: doc["nodes"][0].__setitem__("capacity_slots", 0), "node-a", "capacity_slots"),
            (lambda doc: doc["nodes"][0].__setitem__("lease_limit", 1.5), "node-a", "lease_limit"),
            (lambda doc: doc["nodes"][0].__setitem__("seed", "x"), "node-a", "seed"),
        ]
        for mutate, entry_id, field_name in cases:
            with self.subTest(field=field_name):
                write_sample(self.dir, force=True)
                self.mutate_nodes(mutate)
                with self.assertRaises(SpecError) as ctx:
                    load_bundle(self.dir)
                self.assertEqual(ctx.exception.code, "E_SPEC")
                self.assertEqual(ctx.exception.entry_id, entry_id)
                self.assertEqual(ctx.exception.field, field_name)

    def test_duplicate_artifact_missing_reference_and_cycle(self) -> None:
        self.mutate_pipeline(lambda doc: doc["artifacts"][1].__setitem__("key", "core"))
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self.dir)
        self.assertEqual((ctx.exception.entry_id, ctx.exception.field), ("core", "key"))

        write_sample(self.dir, force=True)
        self.mutate_pipeline(lambda doc: doc["artifacts"][0]["references"].append("missing"))
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self.dir)
        self.assertEqual(ctx.exception.field, "references")

        write_sample(self.dir, force=True)
        self.mutate_pipeline(
            lambda doc: (
                doc["artifacts"][0].__setitem__("references", ["alpha-lib"]),
                doc["artifacts"][1].__setitem__("references", ["core"]),
            )
        )
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self.dir)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_event_sequence_must_increase(self) -> None:
        _, _, events = demo_documents()
        events[1]["seq"] = 1
        self.rewrite_events(events)
        with self.assertRaises(SpecError) as ctx:
            load_bundle(self.dir)
        self.assertEqual(ctx.exception.field, "seq")


if __name__ == "__main__":
    unittest.main()
