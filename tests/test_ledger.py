"""Ledger, snapshot wholesale-replacement and replay tests."""
import os
import tempfile
import unittest

from colval.errors import ReplayError
from colval.ledger import Ledger
from colval.plan import Batch


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="colval-led-")
        self.path = os.path.join(self.dir, "ledger.db")

    def _fill(self, led):
        led.record_batch("t1", Batch(0, 0, 64), "done", 1, "d0", 64,
                         payload="{}")
        led.record_batch("t1", Batch(1, 64, 100), "done", 1, "d1", 100,
                         payload="{}")
        led.record_result("t1", "final", "sum", 42, "(* a b)", 3)
        led.save_snapshot()

    def test_batch_records_atomic_and_cursor(self):
        led = Ledger(self.path)
        self._fill(led)
        recs = led.batch_records()
        self.assertEqual([(r[0], r[1], r[7]) for r in recs],
                         [("t1", 0, 64), ("t1", 1, 100)])
        self.assertEqual(led.done_batches("t1"), {0, 1})
        led.close()

    def test_snapshot_version_match_no_replace(self):
        led = Ledger(self.path)
        self._fill(led)
        led.close()
        led = Ledger(self.path)
        report = led.check_snapshot()
        self.assertFalse(report["replaced"])
        self.assertEqual(len(led.batch_records()), 2)
        led.close()

    def test_snapshot_mismatch_wholesale_replace(self):
        led = Ledger(self.path)
        self._fill(led)
        led.conn.execute("UPDATE snapshot SET snapshot_version=99 WHERE id=1")
        led.conn.commit()
        led.close()
        led = Ledger(self.path)
        report = led.check_snapshot()
        self.assertTrue(report["replaced"])
        self.assertEqual(report["stored_version"], 99)
        self.assertEqual(report["diff"]["batches"], ["t1#0", "t1#1"])
        self.assertEqual(report["diff"]["results"], ["t1"])
        self.assertEqual(led.batch_records(), [])
        self.assertIsNone(led.get_result("t1"))
        log = led.conn.execute("SELECT event FROM recovery_log").fetchall()
        self.assertEqual(log[0][0], "wholesale-replace")
        led.close()

    def test_replay_divergence_raises(self):
        led = Ledger(self.path)
        self._fill(led)
        led.verify_replay("t1", "final")
        with self.assertRaises(ReplayError) as cm:
            led.verify_replay("t1", "tampered")
        self.assertEqual(cm.exception.code, "E_REPLAY")
        self.assertEqual(cm.exception.context["task_id"], "t1")
        led.close()


if __name__ == "__main__":
    unittest.main()
