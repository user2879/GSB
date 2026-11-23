from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gsbcache.arbiter import LeaseArbiter
from gsbcache.errors import ArbiterError
from gsbcache.ledger import Ledger
from gsbcache.processes import NodeCluster
from gsbcache.runner import PipelineRun
from gsbcache.spec import load_spec
from gsbcache.verify import verify_ledger

from tests.helpers import make_sample


def _wait_until_no_active(ledger: Ledger, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not ledger.get_active_leases():
            return
        time.sleep(0.01)
    raise AssertionError("leases did not expire")


class IntegrationTests(unittest.TestCase):
    def test_end_to_end_arbitration_queue_expiry_and_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            run = PipelineRun(
                spec_path,
                root / "work",
                lease_ms=240,
                queue_timeout_ms=5000,
                wait_for_expiry=True,
            )
            result = run.execute()
            self.assertTrue(result["ok"])
            pipeline = result["pipeline"]
            self.assertEqual(len(pipeline["requests"]), 4)
            self.assertTrue(all(item["ok"] for item in pipeline["requests"]))
            self.assertTrue(result["verification"]["ok"])
            self.assertTrue(result["convergence"]["ok"])

            grants = [item for item in pipeline["requests"] if item["ok"]]
            self.assertEqual({item["artifact_key"] for item in grants}, set(spec.artifacts))
            self.assertTrue(all(item["node_id"] in {"node-a", "node-b"} for item in grants))
            self.assertTrue(all("node-c" in {c["node_id"] for c in item["candidates"]} for item in grants))
            for item in grants:
                node_c = next(c for c in item["candidates"] if c["node_id"] == "node-c")
                self.assertFalse(node_c["digest_match"])
                self.assertIn("digest_mismatch", node_c["veto_reason"])
                self.assertNotEqual(node_c["expected_digest"], node_c["actual_digest"])

            queued = [item for item in grants if item["queue_position"] is not None]
            self.assertGreaterEqual(len(queued), 1)
            self.assertIn("lease_limit_exhausted", queued[0]["queue_reason"])
            self.assertIn("capacity_slots_exhausted", queued[0]["queue_reason"])
            self.assertGreaterEqual(queued[0]["queue_ms"], 50)
            self.assertGreaterEqual(queued[0]["total_queue_ms"], queued[0]["queue_ms"])
            for stage_report in pipeline["stages"]:
                self.assertLessEqual(
                    stage_report["observed_stage_inflight_max"],
                    stage_report["concurrency_quota"],
                )
            self.assertLessEqual(
                pipeline["scheduler_metrics"]["global_inflight_max"],
                spec.pipeline.global_concurrency,
            )
            limits = {node.node_id: node.lease_limit for node in spec.nodes}
            for node_id, observed in pipeline["scheduler_metrics"]["node_inflight_max"].items():
                self.assertLessEqual(observed, limits[node_id])

            with Ledger(run.ledger_path, spec, read_only=True) as ledger:
                events = ledger.get_events()
                grant_events = [event for event in events if event["event_type"] == "lease_granted"]
                release_events = [
                    event
                    for event in events
                    if event["event_type"] in {"lease_expired", "quota_recycled", "lease_unknown", "lease_released"}
                ]
                self.assertEqual(len(grant_events), 4)
                self.assertEqual(len(release_events), 4)
                self.assertEqual({event["lease_id"] for event in release_events}, {event["lease_id"] for event in grant_events})
                self.assertEqual({event["details"]["custody_token"] for event in release_events}, {event["lease_id"] for event in grant_events})
                for event in release_events:
                    self.assertEqual(event["details"]["capacity_slots_released"], 1)
                    self.assertEqual(event["details"]["quota_units_released"], 1)
                for node in ledger.get_nodes():
                    self.assertEqual(node["capacity_used"], 0)
                    self.assertEqual(node["capacity_remaining"], node["capacity_slots"])
                    self.assertEqual(node["active_leases"], 0)
                    self.assertEqual(node["quota_remaining"], node["lease_limit"])

    def test_same_key_has_one_owner_and_can_be_reapplied_after_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            ledger_path = root / "ledger.sqlite3"
            ledger = Ledger(ledger_path, spec)
            session = ledger.begin_session(os.getpid())
            cluster = NodeCluster(spec, spec_path, root / "nodes")
            cluster.start_all()
            arbiter = LeaseArbiter(
                spec,
                ledger,
                {node_id: item.client for node_id, item in cluster.nodes.items()},
                session,
                lease_ms=140,
                queue_timeout_ms=3000,
            ).start()
            try:
                def contend(request_id: str):
                    try:
                        return ("granted", arbiter.request_lease("src:main", request_id, stage_id="race"))
                    except ArbiterError as exc:
                        return ("rejected", exc)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(contend, ["race-1", "race-2"]))
                granted = [value for kind, value in results if kind == "granted"]
                rejected = [value for kind, value in results if kind == "rejected"]
                self.assertEqual(len(granted), 1)
                self.assertEqual(len(rejected), 1)
                self.assertEqual(rejected[0].code, "E_ARB")
                self.assertIn(rejected[0].details["reason"], {"key_already_leased", "key_already_reserved"})
                first_lease = granted[0].lease_id
                _wait_until_no_active(ledger)
                second = arbiter.request_lease("src:main", "race-3", stage_id="reapply")
                self.assertTrue(second.ok)
                self.assertNotEqual(first_lease, second.lease_id)
                _wait_until_no_active(ledger)
                active_events = [
                    event
                    for event in ledger.get_events("lease_granted")
                ]
                self.assertEqual(len(active_events), 2)
                self.assertEqual(len({event["lease_id"] for event in active_events}), 2)
            finally:
                arbiter.stop()
                cluster.stop_all()
                ledger.close_session(session, exit_report=True)
                ledger.close()
            self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])

    def test_forced_kill_recovers_only_committed_attempts_and_has_no_exit_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            work = root / "work"
            env = os.environ.copy()
            env["GSBCACHE_PAUSE_AFTER_NODE_GRANT_MS"] = "5000"
            command = [
                sys.executable,
                "-m",
                "gsbcache",
                "run",
                "--spec",
                str(spec_path),
                "--work",
                str(work),
                "--lease-ms",
                "800",
            ]
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            ledger_path = work / "ledger.sqlite3"
            pending = None
            try:
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    if ledger_path.exists():
                        try:
                            conn = sqlite3.connect(str(ledger_path), timeout=0.2)
                            try:
                                rows = conn.execute(
                                    "SELECT attempt_id FROM attempts WHERE state='pending'"
                                ).fetchall()
                            except sqlite3.OperationalError:
                                rows = []
                            finally:
                                conn.close()
                            if rows:
                                pending = [row[0] for row in rows]
                                break
                        except sqlite3.Error:
                            pass
                    if process.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(pending, "did not observe a committed pending attempt")
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
                self.assertNotEqual(process.returncode, 0)
                time.sleep(0.6)
                for node in spec.nodes:
                    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    probe.settimeout(0.2)
                    try:
                        with self.assertRaises(OSError):
                            probe.connect((node.host, node.port))
                    finally:
                        probe.close()

                reopened = Ledger(ledger_path, spec)
                recovery = reopened.recover(current_session=None)
                self.assertEqual(recovery["exit_report"], None)
                self.assertTrue(recovery["requires_reapply"])
                self.assertTrue(set(pending).issubset(set(recovery["unknown_attempts"])))
                sessions = {item["session_id"]: item for item in reopened.get_sessions()}
                self.assertTrue(sessions)
                self.assertTrue(all(item["state"] == "aborted" for item in sessions.values()))
                self.assertTrue(all(item["exit_report"] == 0 for item in sessions.values()))
                reopened.close()
                self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])

                rerun = PipelineRun(
                    spec_path,
                    work,
                    lease_ms=160,
                    queue_timeout_ms=5000,
                    wait_for_expiry=True,
                ).execute()
                self.assertTrue(rerun["ok"])
                self.assertTrue(rerun["verification"]["ok"])
                with Ledger(ledger_path, spec, read_only=True) as final_ledger:
                    sessions = {item["session_id"]: item for item in final_ledger.get_sessions()}
                    self.assertTrue(any(item["state"] == "aborted" and item["exit_report"] == 0 for item in sessions.values()))
                    self.assertTrue(any(item["state"] == "completed" and item["exit_report"] == 1 for item in sessions.values()))
                    attempt_events = final_ledger.get_events("attempt_unknown")
                    self.assertGreaterEqual(len(attempt_events), 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
