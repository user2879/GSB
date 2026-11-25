import json
import os
import subprocess
import sys
import tempfile
import unittest

from batch_receipt_ledger.recovery import recover
from batch_receipt_ledger.spec import parse_spec
from tests._support import make_spec
from tests.test_bootstrap import kill_orphan_workers


class CrashRecoveryE2ETest(unittest.TestCase):
    def test_sigkill_then_recover_only_committed(self):
        kill_orphan_workers()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = os.path.join(tmp.name, "crash.sqlite3")
        raw = make_spec(workers=2, deliveries=9, batch_size=3)
        env = dict(os.environ)
        env["BRL_DEMO_DB"] = db_path
        proc = subprocess.run(
            [sys.executable, "-m", "batch_receipt_ledger.crash_cli"],
            input=json.dumps(raw), env=env, capture_output=True, text=True,
            timeout=30)
        self.assertEqual(proc.returncode, -9)
        kill_orphan_workers()

        report = recover(db_path, parse_spec(raw))
        self.assertEqual(len(report["committed_batches"]), 2)
        self.assertEqual(len(report["unknown_batches"]), 1)
        self.assertTrue(report["replay"]["ok"])
        self.assertTrue(all(item["action"].startswith("quota_reclaimed")
                            for item in report["quota_reclaimed"]))


if __name__ == "__main__":
    unittest.main()
