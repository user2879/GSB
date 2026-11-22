"""Checkpointed, resumable upgrade tests (forced-kill simulation)."""
import json
import os
import tempfile
import unittest

from colval.formats import CURRENT_FORMAT, decode_block, encode_block
from colval.formats import make_bitmap, upgrade_block
from colval.store import BlockStore, block_from_columns


def _blocks():
    return [
        block_from_columns("blk-1", [
            ("a", "int64", make_bitmap([], 3), [1, 2, 3]),
        ], 3, 1024),
        block_from_columns("blk-2", [
            ("b", "float64", make_bitmap([0], 3), [0.0, 1.5, 2.5]),
        ], 3, 1024),
        block_from_columns("blk-3", [
            ("c", "int64", make_bitmap([], 3), [7, 8, 9]),
        ], 3, 1024),
    ]


class TestUpgrade(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="colval-up-")
        store = BlockStore(self.dir)
        for blk, ver in zip(_blocks(), ("b1", "b2", "b3")):
            store.put_block(blk, version=ver)
        store.close()

    def test_upgrade_all_and_backup(self):
        store = BlockStore(self.dir)
        upgraded = store.upgrade_all()
        self.assertEqual(upgraded, ["blk-1", "blk-2"])
        self.assertEqual(sorted(store.list_backups()), ["blk-1", "blk-2"])
        for bid in ("blk-1", "blk-2", "blk-3"):
            self.assertEqual(store.get_block(bid).format_version, CURRENT_FORMAT)
        row = store.conn.execute(
            "SELECT format, payload FROM blocks_backup WHERE block_id='blk-1'"
        ).fetchone()
        self.assertEqual(row[0], "b1")
        self.assertEqual(decode_block(row[1]).format_version, "b1")
        store.close()

    def test_resume_after_kill_no_reprocessing(self):
        store = BlockStore(self.dir)
        first = store.list_block_ids()[0]
        fmt, payload = store.get_payload(first)
        upgraded = upgrade_block(decode_block(payload))
        with store.conn:
            store.conn.execute(
                "INSERT OR IGNORE INTO blocks_backup VALUES(?,?,?,datetime('now'))",
                (first, fmt, payload))
            store.conn.execute(
                "UPDATE blocks SET format=?, payload=? WHERE block_id=?",
                (CURRENT_FORMAT, encode_block(upgraded), first))
        store._write_checkpoint({"current_format": CURRENT_FORMAT,
                                 "upgraded": [first]})
        store.close()
        # restart: must continue from checkpoint, never reprocess blk-1
        store = BlockStore(self.dir)
        seen = []
        upgraded_now = store.upgrade_all(on_upgrade=seen.append)
        self.assertNotIn(first, upgraded_now)
        self.assertEqual(upgraded_now, ["blk-2"])
        self.assertEqual(seen, ["blk-2"])
        self.assertEqual(store.upgrade_all(), [])  # idempotent
        cp = json.load(open(store.checkpoint_path))
        self.assertEqual(cp["upgraded"], ["blk-1", "blk-2", "blk-3"])
        store.close()

    def test_checkpoint_file_atomic(self):
        store = BlockStore(self.dir)
        store.upgrade_all()
        store.close()
        self.assertFalse(os.path.exists(store.checkpoint_path + ".tmp"))


if __name__ == "__main__":
    unittest.main()
