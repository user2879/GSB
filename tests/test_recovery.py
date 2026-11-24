from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from receipt_diag.engine import DiagnosticsEngine
from receipt_diag.errors import DiagnosticError
from receipt_diag.ledger import Ledger
from receipt_diag.specs import load_specs

from tests.helpers import write_sample


ROOT = Path(__file__).resolve().parent.parent


def wait_for_ports_closed(ports: list[int], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        open_ports = []
        for port in ports:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(0.1)
            try:
                connection.connect(("127.0.0.1", port))
                open_ports.append(port)
            except OSError:
                pass
            finally:
                connection.close()
        if not open_ports:
            return
        time.sleep(0.05)


class RecoveryTests(unittest.TestCase):
    def test_forced_termination_recovery_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = write_sample(root / "sample")
            spec = load_specs(sample)
            ports = [worker.port for worker in spec.workers]
            db_path = root / "ledger.db"
            artifacts = root / "artifacts"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "receipt_diag.cli",
                    "run",
                    "--sample-dir",
                    str(sample),
                    "--db",
                    str(db_path),
                    "--artifacts",
                    str(artifacts),
                    "--crash-after-commits",
                    "1",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 137, result.stderr)
            self.assertFalse((artifacts / "run_report.json").exists())
            wait_for_ports_closed(ports)

            ledger = Ledger(db_path)
            try:
                engine = DiagnosticsEngine(spec, artifacts)
                before = engine.recovery_status(ledger)
                self.assertTrue(before["prior_unclean_termination"])
                self.assertFalse(before["can_emit_previous_exit_report"])
                self.assertGreaterEqual(before["completed_count"], 1)
                self.assertLess(before["completed_count"], 5)
                self.assertTrue(before["requires_rerun"])
                unknown = list(before["unknown_batch_ids"])
                recovered = engine.recover(ledger)
                self.assertTrue(recovered["recovery"]["recovered"])
                self.assertEqual(recovered["recovery"]["unknown_batch_ids"], unknown)
                self.assertEqual(recovered["completed_batches"], 5)
                self.assertFalse(recovered["recovery"]["can_emit_previous_exit_report"])
                self.assertEqual(
                    len(recovered["recovery"]["replay_verified_batches"]), 5
                )
                replay = engine.replay(ledger)
                self.assertEqual(len(replay["verified_batches"]), 5)
                after = engine.recovery_status(ledger)
                self.assertEqual(after["session_state"], "completed")
                self.assertFalse(after["requires_rerun"])
            finally:
                ledger.close()

    def test_recover_detects_tampered_committed_output_by_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = write_sample(root / "sample")
            spec = load_specs(sample)
            ports = [worker.port for worker in spec.workers]
            db_path = root / "ledger.db"
            artifacts = root / "artifacts"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "receipt_diag.cli",
                    "run",
                    "--sample-dir",
                    str(sample),
                    "--db",
                    str(db_path),
                    "--artifacts",
                    str(artifacts),
                    "--crash-after-commits",
                    "1",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 137, result.stderr)
            wait_for_ports_closed(ports)
            ledger = Ledger(db_path)
            try:
                engine = DiagnosticsEngine(spec, artifacts)
                committed = engine.recovery_status(ledger)["committed_batch_ids"]
                tampered = committed[0]
                ledger.connection.execute(
                    "UPDATE batches SET output_digest='tampered' WHERE batch_id=?",
                    (tampered,),
                )
                ledger.connection.commit()
                ledger.recompute_content_digest()
                with self.assertRaises(DiagnosticError) as caught:
                    engine.recover(ledger)
                self.assertEqual(caught.exception.code, "E_REPLAY")
                self.assertIn(tampered, caught.exception.path)
                self.assertIn("output_digest", caught.exception.path)
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
