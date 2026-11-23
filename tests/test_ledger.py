"""Ledger persistence, batch atomicity and crash-recovery tests."""

import os
import subprocess
import sys
import tempfile
import unittest

from tests.helpers import make_spec_files
from workbench import UNKNOWN
from workbench.engine import Engine
from workbench.ledger import Ledger
from workbench.spec import load_spec
from workbench.upstream_mgr import UpstreamManager


def _row(seq, **kw):
    row = {"seq": seq, "client_key": "c", "service_id": "svc-a",
           "bucket_key": "alpha", "arrival_ms": 0, "token_before": 0,
           "token_refilled": 0, "allowed": 1, "token_after": 0,
           "auth_ok": 1, "auth_reason": "ok", "decision_code": "OK",
           "receipt_summary": "none", "wait_start_ms": 0, "wait_end_ms": 0,
           "decide_start_ms": 0, "decide_end_ms": 0, "fwd_start_ms": -1,
           "fwd_end_ms": -1}
    row.update(kw)
    return row


class LedgerTest(unittest.TestCase):
    def test_batch_commit_single_transaction(self):
        tmp = tempfile.mkdtemp(prefix="wbled-")
        path = os.path.join(tmp, "ledger.db")
        ledger = Ledger(path, scale=3)
        batch = ledger.begin_batch(1, 2)
        ledger.commit_batch(batch, [_row(1), _row(2)])
        self.assertEqual(ledger.cursor, 2)
        self.assertEqual(len(ledger.done_requests()), 2)
        meta = ledger.meta()
        self.assertEqual(meta["format_version"], "1")
        self.assertEqual(meta["scale"], "3")
        self.assertNotEqual(meta["content_digest"], "0" * 64)
        ledger.close()

    def test_crash_recovery_marks_unknown(self):
        tmp = tempfile.mkdtemp(prefix="wbcrash-")
        path = os.path.join(tmp, "ledger.db")
        ledger = Ledger(path, scale=3)
        batch = ledger.begin_batch(1, 1)
        ledger.commit_batch(batch, [_row(1)])
        ledger.close()
        proc = subprocess.run(
            [sys.executable, "-m", "tests.crash_worker", path],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        ledger2 = Ledger(path)
        unknown = ledger2.recover_uncommitted()
        self.assertEqual(unknown, [101, 102, 103, 104])
        rows = {r["seq"]: r for r in ledger2.iter_requests()}
        self.assertEqual(rows[1]["status"], "done")
        for seq in (101, 102, 103, 104):
            self.assertEqual(rows[seq]["status"], "unknown")
            self.assertEqual(rows[seq]["decision_code"], UNKNOWN)
        ledger2.close()

    def test_engine_run_then_reopen_ledger(self):
        tmp = tempfile.mkdtemp(prefix="wbeng-")
        make_spec_files(tmp)
        spec = load_spec(tmp)
        path = os.path.join(tmp, "ledger.db")
        ledger = Ledger(path, scale=3)
        engine = Engine(spec, ledger, global_limit=2, per_service_quota=1,
                        upstream_timeout_s=1.0)
        with UpstreamManager(spec) as mgr:
            mgr.start()
            engine.run_all(batch_size=3)
        ledger.close()
        ledger2 = Ledger(path)
        self.assertEqual(ledger2.cursor, 8)
        self.assertEqual(len(ledger2.done_requests()), 8)
        ledger2.close()


if __name__ == "__main__":
    unittest.main()
