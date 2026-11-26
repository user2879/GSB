import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shardmesh.canonical import record_digest
from shardmesh.errors import ShardMeshError
from shardmesh.ledger import Ledger
from shardmesh.replay import replay_ledger
from shardmesh.samples import aggregate_batch, generate_batch
from shardmesh.snapshot import load_or_rebuild_snapshot, write_snapshot
from shardmesh.view import build_ledger_view


class LedgerSnapshotReplayTests(unittest.TestCase):
    def test_stale_records_are_excluded_from_current_view(self):
        with TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "stale.sqlite3")
            ledger.commit_outcome(
                {"shard_id": "s1", "node_id": "n1", "outcome": {"summary": {"total": 1}}}
            )
            changed = ledger.flag_records(["s1"], "stale", current=False, stale=True)
            self.assertEqual(changed, ["s1"])
            view = build_ledger_view(ledger)
            self.assertEqual(view["current_count"], 0)
            self.assertEqual(view["stale_count"], 1)
            ledger.close()

    def test_corrupt_snapshot_is_backed_up_and_replaced_whole(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = Ledger(root / "ledger.sqlite3")
            version = ledger.increment_snapshot_version()
            snapshot_path = root / "view.json"
            write_snapshot(snapshot_path, {"snapshot_version": version, "values": [1, 2, 3]})

            payload = json.loads(snapshot_path.read_text())
            payload["values"] = [999]
            snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded, backup = load_or_rebuild_snapshot(
                snapshot_path,
                ledger,
                lambda: {"rebuilt": True, "values": "whole-new-view"},
            )
            self.assertIsNotNone(backup)
            self.assertTrue(backup.exists())
            self.assertEqual((backup.stat().st_mode & 0o777), 0o444)
            self.assertTrue(loaded["rebuilt"])
            ledger.close()

    def test_replay_locates_mismatched_field(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            ledger = Ledger(root / "ledger.sqlite3")
            batch = generate_batch(seed=11, rows=32)
            summary = aggregate_batch(batch, 1000)
            ledger.commit_outcome(
                {
                    "shard_id": "s1",
                    "node_id": "n1",
                    "state": "committed",
                    "outcome": {"summary": summary},
                }
            )
            replay = replay_ledger(ledger, {"batches": {"s1": batch}})
            self.assertEqual(replay["checked_count"], 1)

            row = ledger.conn.execute("SELECT record_json FROM shard_records WHERE shard_id='s1'").fetchone()
            record = json.loads(row["record_json"])
            record["outcome"]["summary"]["total"] += 1
            record["content_digest"] = record_digest(record)
            with ledger.lock, ledger.conn:
                ledger.conn.execute(
                    "UPDATE shard_records SET record_json=?, content_digest=? WHERE shard_id='s1'",
                    (json.dumps(record, sort_keys=True), record["content_digest"]),
                )
            with self.assertRaises(ShardMeshError) as ctx:
                replay_ledger(ledger, {"batches": {"s1": batch}})
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(ctx.exception.field_name, "total")
            self.assertEqual(ctx.exception.details["shard_id"], "s1")
            self.assertEqual(ctx.exception.details["node_id"], "n1")
            ledger.close()


if __name__ == "__main__":
    unittest.main()
