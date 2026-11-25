import tempfile
import unittest
from pathlib import Path

from arbplatform.db import sqlite_version
from arbplatform.determinism import now_ms
from arbplatform.engine import ArbitrationEngine
from arbplatform.errors import ReplayError
from arbplatform.ledger import Ledger
from tests.test_engine import manifest, scalar_set


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "ledger.sqlite"
        self.m = manifest("weighted_majority")

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self, version=1):
        engine = ArbitrationEngine(self.m)
        return engine.arbitrate_view(self.m.view("v"), version, scalar_set())

    def test_sqlite_minimum_version(self):
        major, minor = (int(x) for x in sqlite_version().split(".")[:2])
        self.assertGreaterEqual((major, minor), (3, 40))

    def test_commit_and_replay_match(self):
        ledger = Ledger(self.db, self.m)
        ledger.open_or_replace()
        result = self._result()
        ledger.begin_view("v", 1, now_ms())
        ledger.commit_view(result, now_ms())
        outcome = ledger.replay({"v": result})
        self.assertEqual(outcome.matched, 1)
        ledger.close()

    def test_spec_digest_mismatch_replaces_whole_snapshot(self):
        ledger = Ledger(self.db, self.m)
        ledger.open_or_replace()
        ledger.begin_view("v", 1, now_ms())
        ledger.commit_view(self._result(), now_ms())
        ledger.close()

        changed = manifest("latest_revision")  # different policy -> new digest
        ledger2 = Ledger(self.db, changed)
        outcome = ledger2.open_or_replace()
        self.assertTrue(outcome.replaced)
        self.assertIn("digest", outcome.reason)
        self.assertEqual(ledger2.latest_committed(), {})
        ledger2.close()

    def test_format_version_mismatch_replaces(self):
        ledger = Ledger(self.db, self.m)
        ledger.open_or_replace()
        ledger.begin_view("v", 1, now_ms())
        ledger.commit_view(self._result(), now_ms())
        ledger.conn.execute("UPDATE meta SET format_version = 99 WHERE id = 1")
        ledger.conn.commit()
        ledger.close()

        reopened = Ledger(self.db, self.m)
        outcome = reopened.open_or_replace()
        self.assertTrue(outcome.replaced)
        self.assertIn("format", outcome.reason)
        reopened.close()

    def test_pending_view_marked_unknown_after_kill(self):
        ledger = Ledger(self.db, self.m)
        ledger.open_or_replace()
        ledger.begin_view("v", 7, now_ms())
        # No commit: simulate forced termination while the transaction is open.
        unknown = ledger.mark_unknown_after_kill()
        self.assertEqual(unknown, ["v"])
        rows = ledger.conn.execute(
            "SELECT status FROM views WHERE view_id='v'").fetchall()
        self.assertEqual(rows[0]["status"], "unknown")
        ledger.close()

    def test_replay_locates_first_mismatch(self):
        ledger = Ledger(self.db, self.m)
        ledger.open_or_replace()
        good = self._result()
        ledger.begin_view("v", 1, now_ms())
        ledger.commit_view(good, now_ms())

        tampered = self._result()
        tampered.verdicts["k"].authority_text = "999.99"
        # Keep the view-level envelope identical; mismatch must surface per key.
        with self.assertRaises(ReplayError) as ctx:
            ledger.replay({"v": tampered})
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertIn("views.v.verdicts.k.authority_text", ctx.exception.field)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
