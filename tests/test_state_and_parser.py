import json
import os
import tempfile
import unittest

from syncpipe.errors import E_CONFLICT_UNRESOLVED, E_PEER_GAP, PipelineError
from syncpipe.generator import generate_all
from syncpipe.parser import parse_stream
from syncpipe.state import (
    adjudicate_tick,
    apply_event,
    canonical_bytes,
    initial_state,
    state_digest,
)
from syncpipe.storage import Storage


SEED = 20261119


class DeterminismTests(unittest.TestCase):
    def test_canonical_bytes_order_independent(self):
        a = canonical_bytes({"b": 1, "a": [1, 2, {"z": 0}]})
        b = canonical_bytes({"a": [1, 2, {"z": 0}], "b": 1})
        self.assertEqual(a, b)

    def test_generation_is_deterministic(self):
        first = generate_all(SEED)
        second = generate_all(SEED)
        self.assertEqual(
            canonical_bytes(first),
            canonical_bytes(second),
        )

    def test_parser_orders_by_tick_peer_seq(self):
        streams = generate_all(SEED)
        groups, errors, quarantined = parse_stream(
            json.dumps(e) for e in streams["normal_game"]
        )
        self.assertEqual(errors, [])
        for group in groups:
            keys = [(e["peer_id"], e["local_seq"]) for e in group]
            self.assertEqual(keys, sorted(keys))


class ConflictTests(unittest.TestCase):
    def test_field_lww_records_overridden_writers(self):
        tick = [
            {
                "tick": 1, "peer_id": "peer-a", "local_seq": 0,
                "op_type": "update",
                "payload": {"entity_id": "e1", "fields": {"x": 1}},
            },
            {
                "tick": 1, "peer_id": "peer-b", "local_seq": 0,
                "op_type": "update",
                "payload": {"entity_id": "e1", "fields": {"x": 2}},
            },
        ]
        records = adjudicate_tick(tick)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["winner"]["peer_id"], "peer-b")
        self.assertEqual(records[0]["overridden"][0]["peer_id"], "peer-a")
        self.assertIn("field_lww", records[0]["basis"])

    def test_apply_state_converges_under_lww(self):
        state = initial_state()
        spawn = {
            "tick": 1, "peer_id": "peer-a", "local_seq": 0, "op_type": "spawn",
            "payload": {"entity_id": "e1", "fields": {"x": 0}},
        }
        state = apply_event(state, spawn)
        state = apply_event(state, {
            "tick": 1, "peer_id": "peer-b", "local_seq": 0, "op_type": "update",
            "payload": {"entity_id": "e1", "fields": {"x": 9}},
        })
        self.assertEqual(state["entities"]["e1"]["fields"]["x"], 9)

    def test_missing_entity_is_unresolved(self):
        with self.assertRaises(PipelineError) as ctx:
            apply_event(initial_state(), {
                "tick": 1, "peer_id": "peer-a", "local_seq": 0, "op_type": "move",
                "payload": {"entity_id": "ghost", "fields": {"x": 1}},
            })
        self.assertEqual(ctx.exception.code, E_CONFLICT_UNRESOLVED)

    def test_destroy_tombstones_entity(self):
        state = initial_state()
        state = apply_event(state, {
            "tick": 1, "peer_id": "peer-a", "local_seq": 0, "op_type": "spawn",
            "payload": {"entity_id": "e1", "fields": {}},
        })
        state = apply_event(state, {
            "tick": 2, "peer_id": "peer-a", "local_seq": 1, "op_type": "destroy",
            "payload": {"entity_id": "e1", "fields": {}},
        })
        self.assertNotIn("e1", state["entities"])
        self.assertIn("e1", state["tombstones"])
        with self.assertRaises(PipelineError):
            apply_event(state, {
                "tick": 3, "peer_id": "peer-a", "local_seq": 2, "op_type": "spawn",
                "payload": {"entity_id": "e1", "fields": {}},
            })


class ParserErrorTests(unittest.TestCase):
    def test_illegal_tick_and_gap_and_malformed(self):
        lines = [
            json.dumps({
                "tick": -1, "peer_id": "p", "local_seq": 0, "op_type": "move",
                "payload": {}, "base_snapshot_version": 0,
            }),
            "{not json",
            json.dumps({
                "tick": 1, "peer_id": "p", "local_seq": 5, "op_type": "move",
                "payload": {"entity_id": "e"}, "base_snapshot_version": 0,
            }),
        ]
        groups, errors, quarantined = parse_stream(lines)
        codes = [e["code"] for e in errors]
        self.assertTrue(all(code == E_PEER_GAP for code in codes))
        self.assertEqual(len(quarantined), 3)


class StorageAtomicTests(unittest.TestCase):
    def test_atomic_write_never_leaves_tmp(self):
        with tempfile.TemporaryDirectory() as root:
            storage = Storage(root)
            storage.atomic_write_json("manifest.json", {"a": 1})
            self.assertTrue(os.path.exists(os.path.join(root, "manifest.json")))
            # No half-written temp file is ever left behind.
            self.assertFalse(
                any(name.endswith(".tmp") for name in os.listdir(root))
            )


if __name__ == "__main__":
    unittest.main()
