import os
import sqlite3
import stat
import unittest

from src.ledger import FORMAT_VERSION, Ledger
from tests.util import TmpDir


def _pb(batch_id="batch-0000"):
    return {"batch_id": batch_id, "worker_id": "w0", "param_version": 1,
            "adopted_digest": "a" * 64, "adopted_format": 3,
            "request_ids": ["req-0000", "req-0001"]}


class TestLedger(unittest.TestCase):
    def test_atomic_commit_and_cursor(self):
        with TmpDir() as tmp:
            path = os.path.join(tmp, "l.db")
            ledger = Ledger(path)
            ledger.commit_batch(_pb(), [], [])
            self.assertEqual(ledger.progress_cursor(), 1)
            ledger.commit_batch(_pb("batch-0001"), [], [])
            self.assertEqual(ledger.progress_cursor(), 2)
            batches = ledger.load_batches()
            self.assertEqual(batches["batch-0000"]["status"], "done")
            ledger.close()

    def test_reopen_consistent(self):
        with TmpDir() as tmp:
            path = os.path.join(tmp, "l.db")
            ledger = Ledger(path)
            ledger.commit_batch(_pb(), [], [])
            ledger.close()
            ledger2 = Ledger(path)
            self.assertFalse(ledger2.replaced_on_open)
            self.assertEqual(ledger2.progress_cursor(), 1)
            ledger2.close()

    def test_corruption_replaced_with_readonly_backup(self):
        with TmpDir() as tmp:
            path = os.path.join(tmp, "l.db")
            ledger = Ledger(path)
            ledger.commit_batch(_pb(), [], [])
            ledger.close()
            conn = sqlite3.connect(path)
            conn.execute("UPDATE batches SET adopted_digest='zzz'")
            conn.commit()
            conn.close()
            ledger2 = Ledger(path)
            self.assertTrue(ledger2.replaced_on_open)
            backups = [f for f in os.listdir(tmp) if ".bak-" in f]
            self.assertEqual(len(backups), 1)
            mode = os.stat(os.path.join(tmp, backups[0])).st_mode
            self.assertFalse(mode & stat.S_IWUSR)
            self.assertEqual(ledger2.load_batches(), {})
            ledger2.close()

    def test_format_version_mismatch_replaced(self):
        with TmpDir() as tmp:
            path = os.path.join(tmp, "l.db")
            ledger = Ledger(path)
            ledger.close()
            conn = sqlite3.connect(path)
            conn.execute("UPDATE meta SET value=? WHERE key='format_version'",
                         (str(FORMAT_VERSION + 1),))
            conn.commit()
            conn.close()
            ledger2 = Ledger(path)
            self.assertTrue(ledger2.replaced_on_open)
            self.assertEqual(ledger2.get_meta("format_version"), str(FORMAT_VERSION))
            ledger2.close()

    def test_param_version_registration(self):
        with TmpDir() as tmp:
            ledger = Ledger(os.path.join(tmp, "l.db"))
            v1, changed1, _ = ledger.register_param_version("w0", "digest-a")
            v2, changed2, _ = ledger.register_param_version("w0", "digest-a")
            self.assertEqual((v1, v2), (1, 1))
            self.assertFalse(changed1)
            self.assertFalse(changed2)
            v3, changed3, old = ledger.register_param_version("w0", "digest-b")
            self.assertEqual(v3, 2)
            self.assertTrue(changed3)
            self.assertEqual(old, "digest-a")
            v4, changed4, _ = ledger.register_param_version("w0", "digest-a")
            self.assertEqual(v4, 1)
            self.assertTrue(changed4)
            ledger.close()

    def test_mark_unknown_idempotent(self):
        with TmpDir() as tmp:
            ledger = Ledger(os.path.join(tmp, "l.db"))

            class PB:
                batch_id = "batch-0000"
                worker_id = "w0"
                param_version = 1
                request_ids = ("req-0000",)

            ledger.mark_unknown([PB()])
            ledger.commit_batch(_pb(), [], [])
            ledger.mark_unknown([PB()])
            self.assertEqual(ledger.load_batches()["batch-0000"]["status"], "done")
            ledger.close()

    def test_snapshot_listed(self):
        with TmpDir() as tmp:
            ledger = Ledger(os.path.join(tmp, "l.db"))
            ledger.commit_batch(_pb(), [], [])
            snap = ledger.create_snapshot(tmp)
            self.assertTrue(os.path.exists(snap))
            self.assertIn(os.path.basename(snap), Ledger.list_snapshots(tmp))
            ledger.close()


if __name__ == "__main__":
    unittest.main()
