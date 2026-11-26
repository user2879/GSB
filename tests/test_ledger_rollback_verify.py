import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shard_diag.engine import ExecutionEngine, recover_incomplete_shards
from shard_diag.errors import VerifyError
from shard_diag.ledger import Ledger
from shard_diag.node_manager import NodeManager
from shard_diag.protocol import find_free_port
from shard_diag.spec import Event, EventStream, NodeSpec, RuntimeConfig, ShardSpec, Spec
from shard_diag.verifier import verify_ledger


def rollback_spec():
    node = NodeSpec("n1", "127.0.0.1", find_free_port(), 4, 4, 31337, 1)
    shards = (
        ShardSpec("old", (), 24, 1, 10),
        ShardSpec("new", (), 16, 2, 9),
        ShardSpec("child", ("new",), 16, 1, 1),
    )
    return Spec(1, (node,), shards, 100, RuntimeConfig(lease_ttl_ms=10_000, io_timeout_ms=2_000, batch_size=8))


class LedgerRollbackVerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_rollback_invalidation_unchanged_row_verify_and_tamper_location(self):
        spec = rollback_spec()
        ledger_path = self.tmp / "session.sqlite"
        ledger = Ledger.open_session(ledger_path, spec.digest())
        manager = NodeManager(spec, self.tmp / "nodes")
        manager.start_all()
        engine = ExecutionEngine(spec, ledger, manager, samples_dir=str(self.tmp / "samples"))
        engine.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="test-worker")
        try:
            dispatch_old = Event(1, "dispatch", {"shard_id": "old", "node_id": "n1"})
            engine.handle_event(dispatch_old)
            engine.futures["old"].result(timeout=3)
            before = dict(ledger.get_outcome("old"))

            engine.handle_event(Event(2, "quota_upgrade", {"node_id": "n1", "quota_version": 2}))
            engine.handle_event(Event(3, "dispatch", {
                "shard_id": "new", "node_id": "n1", "processing_delay_ms": 300,
            }))
            rollback_decision = engine.handle_event(Event(4, "rollback", {
                "node_id": "n1", "target_version": 1,
            }))
            invalid = rollback_decision["decision"]["impact"]["invalid_shards"]
            self.assertEqual(invalid, ["new", "child"])
            after = dict(ledger.get_outcome("old"))
            self.assertEqual(before, after)

            engine.pool.shutdown(wait=True)
            events = EventStream((
                dispatch_old,
                Event(2, "quota_upgrade", {"node_id": "n1", "quota_version": 2}),
                Event(3, "dispatch", {"shard_id": "new", "node_id": "n1", "processing_delay_ms": 300}),
                Event(4, "rollback", {"node_id": "n1", "target_version": 1}),
            ))
            report = verify_ledger(spec, events, ledger_path)
            self.assertTrue(report["ok"])

            ledger.close()
            conn = sqlite3.connect(ledger_path)
            conn.execute("UPDATE shard_outcomes SET aggregate_digest=? WHERE shard_id=?", ("deadbeef", "old"))
            conn.commit()
            conn.close()
            with self.assertRaises(VerifyError) as ctx:
                verify_ledger(spec, events, ledger_path)
            self.assertEqual(ctx.exception.code, "E_VERIFY")
            self.assertEqual(ctx.exception.object_id, "old")
            self.assertEqual(ctx.exception.field, "aggregate_digest")
        finally:
            try:
                engine.pool.shutdown(wait=False)
            except Exception:
                pass
            manager.stop_all(force=True)
            ledger.close()

    def test_snapshot_digest_mismatch_replaces_instead_of_merging(self):
        path = self.tmp / "snapshot.sqlite"
        first = Ledger.open_session(path, "digest-a")
        with first.transaction():
            first.set_meta_locked("custom_marker", "keep?")
        first.close()
        replaced = Ledger.open_session(path, "digest-b")
        self.assertTrue(replaced.replaced)
        self.assertEqual(replaced.meta()["spec_digest"], "digest-b")
        self.assertNotIn("custom_marker", replaced.meta())
        replaced.close()

    def test_recovery_marks_granted_without_terminal_outcome_unknown(self):
        spec = rollback_spec()
        ledger = Ledger.open_session(self.tmp / "recover.sqlite", spec.digest())
        with ledger.transaction():
            ledger.journal_event_locked(
                1,
                "hash",
                {"seq": 1, "type": "dispatch", "shard_id": "ghost", "node_id": "n1"},
                {"admitted": True},
            )
            ledger.insert_lease_event_locked({
                "event_seq": 1,
                "lease_id": "L-ghost",
                "shard_id": "ghost",
                "node_id": "n1",
                "kind": "grant",
                "to_version": 1,
            })
        report = recover_incomplete_shards(ledger)
        self.assertEqual(report["unknown_shards"], ["ghost"])
        rows = ledger.recovery_records()
        self.assertEqual(rows[0]["state"], "unknown")
        ledger.close()


if __name__ == "__main__":
    unittest.main()
