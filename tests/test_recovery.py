from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import time
import unittest
from pathlib import Path

from gateway_diag.engine import DiagnosticEngine
from gateway_diag.ledger import LedgerStore
from gateway_diag.model import load_requests, load_spec

from tests.helpers import Workspace, base_spec, reserve_ports, sample_requests


def batch_count(db_path: str) -> int:
    try:
        connection = sqlite3.connect(db_path, timeout=0.2)
        try:
            row = connection.execute("SELECT COUNT(*) FROM batches").fetchone()
            return int(row[0])
        finally:
            connection.close()
    except sqlite3.Error:
        return 0


def wait_ports_closed(ports: list[int], timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        all_closed = True
        for port in ports:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.settimeout(0.05)
            try:
                probe.connect(("127.0.0.1", port))
                all_closed = False
            except OSError:
                pass
            finally:
                probe.close()
        if all_closed:
            return
        time.sleep(0.05)
    raise AssertionError("worker ports did not close after forced termination")


class ForcedTerminationRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Workspace()
        self.ports = reserve_ports(2)
        self.raw_spec = base_spec(self.ports)
        self.raw_spec["limits"]["global_concurrency"] = 1
        self.raw_spec["rules"][0]["batch_size"] = 1
        self.spec_path = self.workspace.write_spec(self.raw_spec)
        self.requests_raw = sample_requests(4)
        self.requests_path = self.workspace.write_requests(self.requests_raw)

    def tearDown(self) -> None:
        self.workspace.close()

    def test_forced_kill_recovers_only_committed_batches_then_replays(self) -> None:
        environment = os.environ.copy()
        environment["GATEWAY_DIAG_TEST_BATCH_DELAY_MS"] = "900"
        command = [
            sys.executable,
            "-m",
            "gateway_diag",
            "run",
            "--spec",
            str(self.spec_path),
            "--requests",
            str(self.requests_path),
            "--db",
            str(self.workspace.db),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 12
            observed = 0
            while time.monotonic() < deadline:
                observed = batch_count(str(self.workspace.db))
                if 1 <= observed < 4:
                    break
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1)
                    self.fail(
                        f"runner exited before forced termination: {process.returncode}\n{stdout}\n{stderr}"
                    )
                time.sleep(0.04)
            else:
                self.fail("no committed batch observed before timeout")
            self.assertGreaterEqual(observed, 1)
            self.assertLess(observed, 4)
            process.kill()
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

        wait_ports_closed(self.ports)
        spec = load_spec(self.spec_path)
        requests = load_requests(self.requests_path, spec)
        with LedgerStore(self.workspace.db) as store:
            engine = DiagnosticEngine(spec, requests, store, startup_timeout=5, receipt_timeout=5)
            plan = engine.plan()
            initial = store.recovery_report(plan, requests)
            self.assertTrue(initial.needs_rerun)
            self.assertGreater(len(initial.unknown_batch_ids), 0)
            self.assertGreater(len(initial.missing_request_ids), 0)
            unknown_rows = store.recovery_unknown_rows(spec.revision)
            self.assertTrue(any(row["status"] == "UNKNOWN" for row in unknown_rows))
            self.assertTrue(initial.ledger_integrity_ok)

            outcome = engine.recover()
            self.assertGreater(outcome.newly_committed, 0)
            self.assertFalse(outcome.final.needs_rerun)
            self.assertTrue(outcome.replay.ledger_integrity_ok)
            self.assertEqual(outcome.replay.replayed_requests, len(self.requests_raw))
            final_unknown_rows = store.recovery_unknown_rows(spec.revision)
            self.assertTrue(
                all(row["status"] == "RESOLVED" for row in final_unknown_rows)
            )
            preview = engine.preview()
            self.assertEqual(len(preview.entries), len(self.requests_raw))
            authority = engine.verify_authoritative(preview)
            self.assertEqual(authority.revision, spec.revision)


if __name__ == "__main__":
    unittest.main()
