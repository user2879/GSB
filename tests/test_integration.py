import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lease_diag.ledger import Ledger
from lease_diag.loader import load_workspace
from lease_diag.runtime import recover_session
from lease_diag.samples import write_samples


class ForcedTerminationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)
        write_samples(self.workdir)

    def tearDown(self):
        self.tmp.cleanup()

    def _cli(self, *extra, expect_rc=0):
        proc = subprocess.run(
            [sys.executable, "-m", "lease_diag",
             "--workdir", str(self.workdir), *extra],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, expect_rc, proc.stdout + proc.stderr)
        return proc.stdout

    def test_open_batch_then_resume_converges_and_replays(self):
        # A real OS process is force-terminated mid-transaction (rc 87).
        self._cli("--hard-exit-batch", "2", "run", expect_rc=87)
        ledger = Ledger(self.workdir / "lease_ledger.sqlite")
        self.assertEqual(ledger.open_batch_ids(), [2])
        self.assertEqual(ledger.progress()["last_request_id"], "R03")
        ledger.close()

        self._cli("--resume", "run")
        ledger = Ledger(self.workdir / "lease_ledger.sqlite")
        self.assertEqual(ledger.open_batch_ids(), [])
        self.assertEqual(
            [row["status"] for row in ledger.list_batches()],
            ["COMMITTED"] * 6)
        ledger.close()

        workspace = load_workspace(self.workdir / "pools.json",
                                   self.workdir / "requests.ndjson")
        result = recover_session(workspace, self.workdir, scale=10000)
        self.assertTrue(result["replay"]["all_match"])
        self.assertEqual(result["replay"]["requests_compared"], 18)
        self.assertEqual(result["recovery"]["unknown_open_batches"], [])
        self.assertEqual(result["ownership"]["active_leases"], 5)

    def test_missing_renewal_reports_e_gap(self):
        self._cli("run")
        gap = self.workdir / "requests_gap.ndjson"
        (self.workdir / "requests.ndjson").write_text(
            gap.read_text(encoding="utf-8"), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "lease_diag",
             "--workdir", str(self.workdir), "recover"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("E_GAP", proc.stdout)
        self.assertIn("R09", proc.stdout)


if __name__ == "__main__":
    unittest.main()
