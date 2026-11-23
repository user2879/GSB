from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cacheproof.content import entry_digest
from cacheproof.spec import load_spec
from cacheproof.store import CacheProofStore


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "cache_spec.json"


class CrashRecoveryTests(unittest.TestCase):
    def test_forced_exit_after_file_replace_marks_unknown(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cacheproof-crash-") as temp:
            workspace = Path(temp) / "workspace"
            with CacheProofStore(
                workspace,
                spec=load_spec(SAMPLE),
                create=True,
                auto_snapshot=False,
            ) as store:
                proof_before = store.build_proof("seg-alpha")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cacheproof",
                    "_crash-worker",
                    "--workspace",
                    str(workspace),
                    "--segment",
                    "seg-alpha",
                    "--key",
                    "alpha/crash",
                    "--size",
                    "24",
                    "--seed",
                    "crash-seed",
                    "--failpoint",
                    "after_file_replace",
                ],
                cwd=str(ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(completed.returncode, 91, completed.stderr)
            with CacheProofStore(workspace, auto_snapshot=False) as store:
                verification = store.verify_all()
                self.assertEqual(verification["code"], "OK")
                self.assertEqual(store.build_proof("seg-alpha"), proof_before)
                self.assertIsNone(
                    store.connection.execute(
                        "SELECT 1 FROM entries WHERE segment_id='seg-alpha' AND entry_key='alpha/crash'"
                    ).fetchone()
                )
                unknown = store.list_unknown_entries()
                self.assertEqual(len(unknown), 1)
                self.assertEqual(unknown[0]["entry_key"], "alpha/crash")
                self.assertEqual(
                    unknown[0]["observed_digest"],
                    entry_digest("alpha/crash", 24, "crash-seed"),
                )
                report = store.recover_unknown_entries()
                self.assertFalse(report["exit_report_available"])
                self.assertEqual(report["recovery_kind"], "clean_open")


if __name__ == "__main__":
    unittest.main()

