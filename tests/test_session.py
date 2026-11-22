"""End-to-end session tests on deterministic local samples."""
import json
import os
import sqlite3
import tempfile
import unittest

from colval.formats import decode_block
from colval.runner import SessionRunner
from colval.sample_data import init_sample_workspace
from colval.store import BlockStore
from colval.verify import verify_store_against_canonical


class TestSession(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="colval-sess-")
        self.paths = init_sample_workspace(self.base)
        self.work = os.path.join(self.base, "work")

    def _run(self, crash_after=None):
        runner = SessionRunner(self.paths["data_dir"], self.paths["tasks"],
                               self.work)
        try:
            return runner.run(crash_after=crash_after)
        finally:
            runner.close()

    def test_full_session_and_verify(self):
        report = self._run()
        self.assertFalse(report["crashed"])
        self.assertEqual(report["upgraded"],
                         ["blk-a1", "blk-a2", "blk-b1", "blk-b2"])
        with open(self.paths["canonical"], encoding="utf-8") as fh:
            canonical = {b: decode_block(p) for b, p in json.load(fh).items()}
        store = BlockStore(self.paths["data_dir"])
        verified = verify_store_against_canonical(store, canonical)
        store.close()
        self.assertEqual(len(verified), 6)
        self.assertLessEqual(report["scheduler"]["max_running_units"], 6)
        quotas = {"t1": 2, "t2": 2, "t3": 3, "t4": 1, "t5": 2}
        for tid, peak in report["scheduler"]["max_running_per_task"].items():
            self.assertLessEqual(peak, quotas[tid])
        for tm in report["scheduler"]["timings"]:
            self.assertIsNotNone(tm["queue_wait_ms"])
            self.assertGreater(tm["peak_mem_bytes"], 0)
        self.assertGreater(report["tasks"]["t4"]["div_zero_rows"], 0)

    def test_crash_resume_byte_identical(self):
        ref = self._run()
        ref_digests = {t: v["final_digest"] for t, v in ref["tasks"].items()}
        base2 = tempfile.mkdtemp(prefix="colval-sess2-")
        paths2 = init_sample_workspace(base2)
        work2 = os.path.join(base2, "work")
        runner = SessionRunner(paths2["data_dir"], paths2["tasks"], work2)
        crash = runner.run(crash_after=7)
        runner.close()
        self.assertTrue(crash["crashed"])
        self.assertEqual(crash["committed_batches"], 7)
        runner = SessionRunner(paths2["data_dir"], paths2["tasks"], work2)
        resumed = runner.run()
        runner.close()
        self.assertEqual(ref_digests, {t: v["final_digest"]
                                       for t, v in resumed["tasks"].items()})

    def test_snapshot_mismatch_replaces_then_matches(self):
        self._run()
        conn = sqlite3.connect(os.path.join(self.work, "ledger.db"))
        conn.execute("UPDATE snapshot SET snapshot_version=1 WHERE id=1")
        conn.commit()
        conn.close()
        report = self._run()
        self.assertTrue(report["snapshot"]["replaced"])
        report2 = self._run()
        self.assertFalse(report2["snapshot"]["replaced"])


if __name__ == "__main__":
    unittest.main()
