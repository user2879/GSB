"""Requirements 6/8: cached view vs authoritative replay, E_DIVERGE, ledger."""
import os
import tempfile
import unittest

from shardmeta.clock import ManualClock
from shardmeta.engine import AccountingEngine
from shardmeta.errors import DivergenceError
from shardmeta.ledger import Ledger
from shardmeta.models import (
    NodeSpec,
    ObjectManifest,
    Request,
    ShardPlacement,
)


def build_world():
    nodes = {
        "n1": NodeSpec("n1", "127.0.0.1", 44001, 4, 3, 1, 1),
        "n2": NodeSpec("n2", "127.0.0.1", 44002, 4, 3, 2, 1),
    }
    manifests = {
        "obj": ObjectManifest(
            object_key="obj", total_bytes=16, shard_size=8, content_seed=5,
            placements=(
                ShardPlacement(0, "n1", 8),
                ShardPlacement(1, "n2", 8),
            ),
        )
    }
    return nodes, manifests


class EngineLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmp, "ledger.sqlite3")
        self.clock = ManualClock(1_000_000)
        nodes, manifests = build_world()
        self.nodes, self.manifests = nodes, manifests
        self.ledger = Ledger(self.ledger_path)
        self.engine = AccountingEngine(
            nodes, manifests, self.ledger, orchestrator=None,
            clock=self.clock, lease_ttl_ms=100,
        )

    def tearDown(self):
        self.ledger.close()

    def test_fast_read_uses_cache_and_matches_authoritative_replay(self):
        self.engine.process_request(Request(1, "grant", "obj", 0, ttl_ms=100))
        self.engine.process_request(Request(2, "upload", "obj", 0))
        self.engine.process_request(Request(3, "upload", "obj", 1))
        read = self.engine.fast_read("obj")
        self.assertTrue(read["served_from_cache"])
        self.assertEqual(len(read["view"]["adopted"]), 2)
        # Cache conclusion equals authoritative ledger recompute.
        self.engine.verify_against_ledger()
        replay = self.engine.authoritative_replay()
        self.assertEqual(set(replay["obj"][0].winner.__dict__),
                         set(self.engine.adoption["obj"][0].winner.__dict__))

    def test_divergence_locates_first_key_shard_field(self):
        self.engine.process_request(Request(1, "upload", "obj", 0))
        # Tamper with the in-memory cached candidate digest only.
        candidate = self.engine._candidates[0]
        # MergeCandidate is frozen; replace it with a corrupted copy.
        from dataclasses import replace
        corrupted = replace(candidate, shard_digest="f" * 16)
        self.engine._candidates[0] = corrupted
        self.engine._rebuild_adoption()
        with self.assertRaises(DivergenceError) as ctx:
            self.engine.verify_against_ledger()
        self.assertEqual(ctx.exception.code, "E_DIVERGE")
        self.assertEqual(ctx.exception.key, "obj")
        self.assertEqual(ctx.exception.field, "shard_digest")
        self.assertEqual(ctx.exception.detail["shard_index"], 0)

    def test_atomic_commit_writes_event_lease_counter_cursor_together(self):
        self.engine.process_request(Request(1, "grant", "obj", 0, ttl_ms=100))
        events = self.ledger.fetch_events()
        self.assertEqual(events[0]["op"], "grant")
        counters = {row["node_id"]: dict(row)
                    for row in self.ledger.fetch_counters()}
        self.assertEqual(counters["n1"]["active_count"], 1)
        self.assertEqual(counters["n1"]["remaining_slots"], 3)
        self.assertGreaterEqual(self.ledger.progress_cursor(), 1)
        # Digest metadata is maintained.
        self.ledger.verify_digest()

    def test_snapshot_version_mismatch_replaces_wholesale(self):
        self.engine.process_request(Request(1, "upload", "obj", 0))
        digest = self.ledger.refresh_digest()
        self.ledger.close()
        opened_ok = Ledger.open_or_replace(self.ledger_path, expected_digest=digest)
        self.assertFalse(opened_ok.open_result.replaced)
        opened_ok.close()

        # Corrupt the stored format version to force replacement.
        import sqlite3
        con = sqlite3.connect(self.ledger_path)
        con.execute("UPDATE ledger_meta SET value='999' WHERE key='format_version'")
        con.commit()
        con.close()
        opened = Ledger.open_or_replace(self.ledger_path, expected_digest=digest)
        self.assertTrue(opened.open_result.replaced)
        self.assertIn("format-version", opened.open_result.reason)
        opened.close()
        fresh = Ledger(self.ledger_path)
        self.assertEqual(fresh.fetch_events(), [])
        fresh.close()

    def test_restore_rebuilds_only_from_committed_rows(self):
        self.engine.process_request(Request(1, "grant", "obj", 0, ttl_ms=100))
        self.engine.process_request(Request(2, "upload", "obj", 0))
        state = self.engine.restore_state_from_ledger()
        self.assertEqual(state["leases"], 1)
        self.assertEqual(state["candidates"], 1)
        self.engine.verify_against_ledger()


if __name__ == "__main__":
    unittest.main()
