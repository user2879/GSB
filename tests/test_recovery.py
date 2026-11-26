from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from gsb.runner import run_spec
from gsb.verify import verify_ledger

from tests.factories import free_tcp_port, write_spec


ROOT = Path(__file__).resolve().parent.parent


def crash_spec() -> dict:
    return {
        "session_id": "crash-test",
        "scale_factor": 100,
        "budget": {"resource_units": 10},
        "scheduling": {
            "lease_ttl_ms": 30_000,
            "request_timeout_ms": 10_000,
            "handshake_timeout_ms": 10_000,
            "poll_interval_ms": 5,
        },
        "nodes": [
            {
                "node_id": "slow",
                "host": "127.0.0.1",
                "port": free_tcp_port(),
                "concurrency": 1,
                "lease_limit": 1,
                "seed": 1001,
                "capacity_version": 4,
                "processing_delay_ms": 2_000,
            },
            {
                "node_id": "fast",
                "host": "127.0.0.1",
                "port": free_tcp_port(),
                "concurrency": 1,
                "lease_limit": 1,
                "seed": 1002,
                "capacity_version": 4,
                "processing_delay_ms": 5,
            },
        ],
        "shards": [
            {
                "shard_id": "slow-shard",
                "node_id": "slow",
                "dependencies": [],
                "rows": 16,
                "resource_units": 3,
                "priority": 100,
            },
            {
                "shard_id": "fast-shard",
                "node_id": "fast",
                "dependencies": [],
                "rows": 16,
                "resource_units": 1,
                "priority": 1,
            },
        ],
    }


def _db_state(db_path: Path) -> tuple[set[str], list[dict]]:
    if not db_path.exists():
        return set(), []
    try:
        with sqlite3.connect(db_path) as conn:
            outcomes = {row[0] for row in conn.execute("SELECT shard_id FROM outcomes")}
            leases = [
                {"lease_id": row[0], "shard_id": row[1], "status": row[2], "units": row[3]}
                for row in conn.execute(
                    "SELECT lease_id, shard_id, status, resource_units FROM lease_records"
                )
            ]
        return outcomes, leases
    except sqlite3.Error:
        return set(), []


def _kill_node_groups(log_dir: Path) -> None:
    for pid_file in log_dir.glob("node-*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                os.kill(pid, signal.SIGKILL)
        except (OSError, ValueError):
            pass


class CrashRecoveryTests(unittest.TestCase):
    def test_sigkill_recovery_and_independent_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = write_spec(root / "crash.json", crash_spec())
            db_path = root / "crash.db"
            log_dir = root / "logs"
            log_dir.mkdir()
            parent_log = (root / "parent.log").open("wb")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gsb",
                    "run",
                    str(spec_path),
                    "--db",
                    str(db_path),
                    "--log-dir",
                    str(log_dir),
                    "--run-id",
                    "crash",
                ],
                cwd=str(ROOT),
                env=env,
                stdout=parent_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            killed = False
            try:
                deadline = time.monotonic() + 15
                ready = False
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        detail = (root / "parent.log").read_text(encoding="utf-8", errors="replace")
                        self.fail(f"run process exited early: {detail!r}")
                    outcomes, leases = _db_state(db_path)
                    active_slow = [
                        lease
                        for lease in leases
                        if lease["shard_id"] == "slow-shard" and lease["status"] in {"granted", "renewed"}
                    ]
                    if "fast-shard" in outcomes and active_slow:
                        ready = True
                        break
                    time.sleep(0.05)
                self.assertTrue(ready, "did not observe committed fast shard and active slow lease")

                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                _kill_node_groups(log_dir)
                killed = True
                time.sleep(0.2)

                recover = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "gsb",
                        "recover",
                        "--spec",
                        str(spec_path),
                        "--db",
                        str(db_path),
                    ],
                    cwd=str(ROOT),
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(recover.returncode, 0, recover.stderr + recover.stdout)
                report = json.loads(recover.stdout)
                self.assertGreaterEqual(report["reclaimed_leases"], 1)
                self.assertGreaterEqual(report["reclaimed_units"], 3)
                self.assertIn("slow-shard", report["unknown_shards"])
                self.assertNotIn("fast-shard", report["unknown_shards"])

                rerun = run_spec(
                    spec_path,
                    db_path,
                    log_dir=root / "rerun-logs",
                    run_id="recovery-rerun",
                )
                self.assertEqual(rerun["result"]["status"], "completed")
                verification = verify_ledger(spec_path, db_path)
                self.assertEqual(verification["verified_shards"], 2)
            finally:
                parent_log.close()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                if not killed:
                    _kill_node_groups(log_dir)


if __name__ == "__main__":
    unittest.main()
