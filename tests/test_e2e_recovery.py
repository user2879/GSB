from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lease_arbitrator.errors import VerificationError
from lease_arbitrator.runner import SampleRunner
from lease_arbitrator.sample import write_sample
from lease_arbitrator.ui import build_view_model, render_text_snapshot
from lease_arbitrator.verify import verify_ledger


ROOT = Path(__file__).resolve().parents[1]


def free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def write_free_port_sample(directory: Path) -> None:
    write_sample(directory)
    path = directory / "nodes.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for node in doc["nodes"]:
        node["port"] = free_tcp_port()
    path.write_text(json.dumps(doc), encoding="utf-8")


class EndToEndTests(unittest.TestCase):
    def test_sample_arbitration_rollback_watermark_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            state = root / "state"
            write_free_port_sample(sample)

            with SampleRunner(sample, state) as runner:
                report = runner.run()

            self.assertEqual(report["last_event_seq"], 12)
            outcomes = {item["seq"]: item for item in report["outcomes"]}
            self.assertEqual(outcomes[1]["winner"], "node-c")
            self.assertEqual(outcomes[2]["winner"], "node-a")
            self.assertEqual(outcomes[3]["winner"], "node-b")
            self.assertEqual(outcomes[4]["outcome"], "rolled_back")
            self.assertEqual(outcomes[7]["code"], "E_LEASE")
            self.assertEqual(outcomes[9]["expired"], 1)
            self.assertEqual(outcomes[11]["revoked"], 2)

            verification = verify_ledger(sample, state / "ledger.sqlite")
            self.assertTrue(verification["ok"])

            conn = sqlite3.connect(state / "ledger.sqlite")
            conn.row_factory = sqlite3.Row
            reserve = conn.execute(
                "SELECT payload_json FROM records WHERE event_seq=4 AND record_type='reserve'"
            ).fetchone()
            rollback = conn.execute(
                "SELECT payload_json FROM records WHERE event_seq=4 AND record_type='rollback'"
            ).fetchone()
            self.assertEqual(json.loads(reserve[0])["watermark"]["watermark_fixed"], 1000)
            self.assertEqual(json.loads(rollback[0])["watermark_after_rollback"]["watermark_fixed"], 500)
            conn.close()

            text = render_text_snapshot(build_view_model(state / "ledger.sqlite"))
            self.assertIn("node-a,node-b", text)
            self.assertIn("outcome=rollback", text)
            self.assertIn("Unknown: 0 open", text)

            corrupt = sqlite3.connect(state / "ledger.sqlite")
            corrupt.execute("UPDATE node_state SET used_slots=used_slots+1 WHERE node_id='node-a'")
            corrupt.commit()
            corrupt.close()
            with self.assertRaises(VerificationError) as mismatch:
                verify_ledger(sample, state / "ledger.sqlite")
            self.assertEqual(mismatch.exception.code, "E_VERIFY")
            self.assertEqual(mismatch.exception.details["node_id"], "node-a")
            self.assertIn(mismatch.exception.field, {"used_slots", "remaining_slots"})

    def test_force_kill_unknown_recover_and_replay_verifies(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        python = sys.executable
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            state = root / "state"
            write_free_port_sample(sample)
            base_command = [python, "-m", "lease_arbitrator"]

            first = subprocess.run(
                [*base_command, "run", "--sample", str(sample), "--state-dir", str(state), "--crash-batch", "4"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            self.assertEqual(first.returncode, 97, msg=first.stdout + first.stderr)
            self.assertFalse((state / "run-report.json").exists())

            recover = subprocess.run(
                [*base_command, "recover", "--sample", str(sample), "--state-dir", str(state)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(recover.returncode, 0, msg=recover.stdout + recover.stderr)

            with self.assertRaises(VerificationError) as ctx:
                verify_ledger(sample, state / "ledger.sqlite")
            self.assertEqual(ctx.exception.code, "E_UNKNOWN")

            second = subprocess.run(
                [*base_command, "run", "--sample", str(sample), "--state-dir", str(state)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            self.assertEqual(second.returncode, 0, msg=second.stdout + second.stderr)
            report = json.loads(second.stdout)
            outcomes = {item["seq"]: item for item in report["outcomes"]}
            self.assertEqual(outcomes[4]["outcome"], "rolled_back")
            self.assertEqual(outcomes[6]["outcome"], "renewed")
            self.assertEqual(outcomes[7]["outcome"], "renew_rejected")

            verification = verify_ledger(sample, state / "ledger.sqlite")
            self.assertTrue(verification["ok"])

    def test_topology_replacement_revokes_and_restarts_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            state = root / "state"
            write_free_port_sample(sample)
            events = [
                {"seq": 1, "type": "write", "artifact_key": "core", "lease_ref": "core-v1", "lease_ms": 60000},
                {"seq": 2, "type": "quota_change", "quota_version": 2, "active_nodes": ["node-a", "node-b"]},
                {"seq": 3, "type": "quota_change", "quota_version": 3, "active_nodes": ["node-a", "node-b", "node-c"]},
                {"seq": 4, "type": "write", "artifact_key": "core", "lease_ref": "core-v3", "lease_ms": 60000},
            ]
            (sample / "events.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")

            with SampleRunner(sample, state) as runner:
                report = runner.run()
            outcomes = {item["seq"]: item for item in report["outcomes"]}
            self.assertEqual(outcomes[1]["winner"], "node-c")
            self.assertEqual(outcomes[2]["removed"], ["node-c"])
            self.assertEqual(outcomes[2]["revoked"], 1)
            self.assertEqual(outcomes[3]["added"], ["node-c"])
            self.assertEqual(outcomes[4]["winner"], "node-c")
            self.assertTrue(verify_ledger(sample, state / "ledger.sqlite")["ok"])

    def test_cancel_after_reserve_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            state = root / "state"
            write_free_port_sample(sample)
            events = [
                {
                    "seq": 1,
                    "type": "write",
                    "artifact_key": "core",
                    "candidate_nodes": ["node-a", "node-b"],
                    "lease_ref": "cancelled",
                    "lease_ms": 60000,
                    "cancel_after_reserve": True,
                }
            ]
            (sample / "events.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")
            with SampleRunner(sample, state) as runner:
                report = runner.run()
            self.assertEqual(report["outcomes"][0]["outcome"], "cancelled_rolled_back")
            verification = verify_ledger(sample, state / "ledger.sqlite")
            self.assertTrue(verification["ok"])


if __name__ == "__main__":
    unittest.main()
