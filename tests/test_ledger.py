import os
import tempfile
import unittest

from leasebatch.engine import RunEngine
from leasebatch.errors import ReplayError
from leasebatch.ledger import FORMAT_VERSION, Ledger
from leasebatch.plan import build_plan
from leasebatch.spec import load_spec_dir


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec_dir(
            os.path.join(os.path.dirname(__file__), "..", "samples")
        )
        self.plan = build_plan(self.spec, 2, 42)

    def test_snapshot_replace_is_whole_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger.db")
            ledger = Ledger(path)
            first = ledger.replace_snapshot(self.plan, 1)
            second = ledger.replace_snapshot(self.plan, 2)
            self.assertEqual(second, first + 1)
            rows = ledger.conn.execute(
                "SELECT COUNT(*) FROM snapshots"
            ).fetchone()[0]
            self.assertEqual(rows, 1)
            ledger.close()

    def test_commit_is_atomic_and_replay_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger.db")
            engine = RunEngine(self.spec, path)
            out = engine.execute(2, 42)
            ledger = Ledger(path)
            self.assertTrue(ledger.replay_verify(out["expected"]))
            # Tamper one field and confirm E_REPLAY locates it.
            bad = [dict(item) for item in out["expected"]]
            bad[0] = dict(bad[0], progress_cursor=999)
            with self.assertRaises(ReplayError) as caught:
                ledger.replay_verify(bad)
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertEqual(caught.exception.field, "progress_cursor")
            ledger.close()

    def test_forced_termination_leaves_unknown_and_reclaimable_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger.db")
            engine = RunEngine(self.spec, path)
            crash_id = self.plan.batches[0].batch_id
            out = engine.execute(2, 42, in_flight_batch_ids={crash_id})
            recovery = out["recovery"]
            self.assertEqual(
                [row["batch_id"] for row in recovery["unknown"]], [crash_id]
            )
            # Only committed rows are trusted for replay.
            self.assertEqual(
                len(recovery["committed"]), len(out["expected"])
            )
            ledger = Ledger(path)
            self.assertTrue(ledger.replay_verify(out["expected"]))
            ledger.close()

    def test_records_carry_integer_format_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ledger.db")
            engine = RunEngine(self.spec, path)
            engine.execute(2, 42)
            ledger = Ledger(path)
            versions = ledger.conn.execute(
                "SELECT DISTINCT format_version FROM batches"
            ).fetchall()
            self.assertEqual(versions, [(FORMAT_VERSION,)])
            ledger.close()


if __name__ == "__main__":
    unittest.main()
