from __future__ import annotations

import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from leasemesh.cli import main as cli_main
from leasemesh.cluster import NodeCluster, allocate_ports, port_is_open
from leasemesh.errors import MeshError, VerifyError
from leasemesh.snapshot import reconcile_snapshot
from leasemesh.spec import digest_declaration, load_declaration, load_events
from leasemesh.store import Ledger
from leasemesh.verify import verify_ledger


ROOT = Path(__file__).resolve().parents[1]


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        source_spec = load_declaration(ROOT / "samples/declarations.json")
        self.declaration_path = self.work / "declarations.runtime.json"
        allocate_ports(source_spec, self.declaration_path)
        self.spec = load_declaration(self.declaration_path)
        self.ledger_path = self.work / "ledger.sqlite3"
        self.snapshot_path = self.work / "snapshot.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_full_live_mesh_verify_snapshot_and_tamper_detection(self) -> None:
        exit_code = cli_main(
            [
                "execute",
                "--declarations",
                str(self.declaration_path),
                "--requests",
                str(ROOT / "samples/requests.json"),
                "--workdir",
                str(self.work / "cluster"),
                "--ledger",
                str(self.ledger_path),
                "--snapshot",
                str(self.snapshot_path),
                "--reset",
            ]
        )
        self.assertEqual(0, exit_code)
        verification = verify_ledger(self.ledger_path, self.declaration_path)
        self.assertTrue(verification["ok"])
        self.assertEqual(26, verification["checked_commits"])
        self.assertEqual(13, verification["replayed_events"])

        conn = sqlite3.connect(self.ledger_path)
        generations = dict(
            conn.execute("SELECT lease_id, generation FROM leases ORDER BY lease_id").fetchall()
        )
        wait_rows = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE status='committed' AND wait_ms IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(3, generations["lease-a1"])
        self.assertEqual(2, generations["lease-b1"])
        self.assertEqual(1, generations["lease-c2"])
        self.assertEqual(13, wait_rows)

        changed_spec = copy.deepcopy(self.spec)
        changed_spec["quota_version"] = 99
        changed_spec["declaration_digest"] = digest_declaration(changed_spec)
        ledger = Ledger(self.ledger_path, changed_spec)
        replacement = reconcile_snapshot(self.snapshot_path, changed_spec, ledger)
        ledger.close()
        self.assertEqual("replaced_wholesale", replacement["action"])
        self.assertIsNotNone(replacement["backup"])
        backup = Path(replacement["backup"])
        self.assertEqual(0o444, backup.stat().st_mode & 0o777)
        self.assertTrue(Path(replacement["diff_path"]).exists())

        tamper_conn = sqlite3.connect(self.ledger_path)
        tamper_conn.execute("UPDATE ownership SET node_id='node-tampered'")
        tamper_conn.commit()
        tamper_conn.close()
        with self.assertRaises(VerifyError) as caught:
            verify_ledger(self.ledger_path, self.declaration_path)
        self.assertEqual("E_VERIFY", caught.exception.code)
        self.assertIsNotNone(caught.exception.field)

    def test_stale_generation_is_rejected_without_advance(self) -> None:
        cluster = NodeCluster(self.spec, self.work / "stale-cluster")
        cluster.start()
        try:
            node_id = self.spec["nodes"][0]["node_id"]
            grant = cluster.request(
                node_id,
                {"op": "grant", "request_id": "g1", "lease_id": "lease-x", "slot_id": "slot-1"},
            )
            self.assertEqual(1, grant["generation"])
            renewed = cluster.request(
                node_id,
                {
                    "op": "renew",
                    "request_id": "r1",
                    "lease_id": "lease-x",
                    "generation": 1,
                    "label": "lease-x@1",
                },
            )
            self.assertEqual(2, renewed["generation"])
            with self.assertRaises(MeshError) as caught:
                cluster.request(
                    node_id,
                    {
                        "op": "renew",
                        "request_id": "r2-stale",
                        "lease_id": "lease-x",
                        "generation": 1,
                        "label": "lease-x@1",
                    },
                )
            self.assertEqual("E_LEASE", caught.exception.code)
            self.assertEqual(2, caught.exception.details["generation"])
        finally:
            cluster.stop()

    def test_sigkill_recovery_and_independent_replay(self) -> None:
        command = [
            sys.executable,
            "-m",
            "leasemesh",
            "execute",
            "--declarations",
            str(self.declaration_path),
            "--requests",
            str(ROOT / "samples/requests-crash.json"),
            "--workdir",
            str(self.work / "crash-cluster"),
            "--ledger",
            str(self.ledger_path),
            "--snapshot",
            str(self.snapshot_path),
            "--reset",
            "--crash-before-commit",
            "req-crash-grant",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(-9, completed.returncode, completed.stdout + completed.stderr)

        node_port = self.spec["nodes"][1]["port"]
        self.assertTrue(
            _wait_port_closed(node_port),
            "node subprocess survived parent SIGKILL",
        )

        ledger = Ledger(self.ledger_path, self.spec)
        recovery = ledger.recover_pending()
        status = ledger.conn.execute(
            "SELECT status, recovery_state FROM requests WHERE request_id='req-crash-grant'"
        ).fetchone()
        reservation = ledger.conn.execute(
            "SELECT status FROM reservations WHERE request_id='req-crash-grant'"
        ).fetchone()
        ledger.close()

        self.assertEqual(1, recovery["count"])
        self.assertEqual("unknown", status[0])
        self.assertEqual("rerun_required", status[1])
        self.assertEqual("reclaimed", reservation[0])
        verification = verify_ledger(self.ledger_path, self.declaration_path)
        self.assertEqual(2, verification["checked_commits"])
        self.assertEqual(0, verification["replayed_events"])

    def test_event_stream_import_replays_to_same_decisions(self) -> None:
        events = load_events(ROOT / "samples/events.json")
        event_ledger = self.work / "events.sqlite3"
        ledger = Ledger(event_ledger, self.spec)
        ledger.import_events(events)
        ledger.close()
        verification = verify_ledger(event_ledger, self.declaration_path)
        self.assertTrue(verification["ok"])
        self.assertEqual(11, verification["replayed_events"])
        self.assertEqual(2, verification["decisions"])


def _wait_port_closed(port: int, timeout: float = 4.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_is_open("127.0.0.1", port, timeout_ms=100):
            return True
        time.sleep(0.05)
    return not port_is_open("127.0.0.1", port, timeout_ms=100)


if __name__ == "__main__":
    unittest.main()
