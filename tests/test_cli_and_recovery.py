from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.helpers import ROOT, behavior, free_ports, instance, make_spec, write_json


def run_cli(*args: str, timeout: float = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "meshwork.cli", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class CliAnomalyTests(unittest.TestCase):
    def test_three_injections_return_their_normal_business_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = [
                "run",
                "--spec",
                str(ROOT / "samples" / "sample_spec.json"),
                "--requests",
                str(ROOT / "samples" / "requests.json"),
            ]
            spec_result = run_cli(
                *common,
                "--ledger",
                str(root / "spec.db"),
                "--reset",
                "--inject",
                "corrupt-rules",
                "--log-file",
                str(root / "spec.jsonl"),
            )
            self.assertEqual(spec_result.returncode, 2)
            self.assertEqual(json.loads(spec_result.stderr)["code"], "E_SPEC")

            port_result = run_cli(
                *common,
                "--ledger",
                str(root / "port.db"),
                "--reset",
                "--inject",
                "port-occupied",
                "--log-file",
                str(root / "port.jsonl"),
            )
            self.assertEqual(port_result.returncode, 2)
            self.assertEqual(json.loads(port_result.stderr)["code"], "E_PORT")

            snapshot_result = run_cli(
                *common,
                "--ledger",
                str(root / "snapshot.db"),
                "--snapshot-dir",
                str(root / "snapshots"),
                "--reset",
                "--inject",
                "snapshot-digest",
                "--log-file",
                str(root / "snapshot.jsonl"),
            )
            self.assertEqual(snapshot_result.returncode, 2)
            self.assertEqual(json.loads(snapshot_result.stderr)["code"], "E_SNAPSHOT")

            events = []
            for path in (root / "spec.jsonl", root / "port.jsonl", root / "snapshot.jsonl"):
                events.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            codes = {event.get("code") for event in events}
            self.assertTrue({"E_SPEC", "E_PORT", "E_SNAPSHOT"}.issubset(codes))


class ForcedTerminationRecoveryTest(unittest.TestCase):
    def test_kill_restart_marks_unknown_and_reruns_from_committed_ledger(self) -> None:
        ports = free_ports(2)
        raw = make_spec(
            [
                {
                    "id": "slow",
                    "quota": 2,
                    "instances": [
                        instance(
                            "slow-a",
                            port=ports[0],
                            weight=1,
                            behavior_value=behavior(
                                fast_ms=120,
                                slow_ms=150,
                                rules=[
                                    {
                                        "request_prefix": "crash-",
                                        "outcome": "success",
                                        "latency_class": "fast",
                                    }
                                ],
                            ),
                        ),
                        instance(
                            "slow-b",
                            port=ports[1],
                            weight=1,
                            behavior_value=behavior(
                                fast_ms=120,
                                slow_ms=150,
                                rules=[
                                    {
                                        "request_prefix": "crash-",
                                        "outcome": "success",
                                        "latency_class": "fast",
                                    }
                                ],
                            ),
                        ),
                    ],
                }
            ],
            global_concurrency=2,
            timeout_ms=500,
            backoffs=[2],
        )
        requests = {
            "requests": [
                {"id": f"crash-{index:03d}", "service": "slow", "payload": {"n": index}}
                for index in range(20)
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec_path = write_json(root / "spec.json", raw)
            requests_path = write_json(root / "requests.json", requests)
            ledger_path = root / "ledger.db"
            snapshot_dir = root / "snapshots"
            command = [
                sys.executable,
                "-m",
                "meshwork.cli",
                "run",
                "--spec",
                str(spec_path),
                "--requests",
                str(requests_path),
                "--ledger",
                str(ledger_path),
                "--snapshot-dir",
                str(snapshot_dir),
                "--reset",
                "--log-file",
                str(root / "crash.jsonl"),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            intents = 0
            for _ in range(250):
                time.sleep(0.02)
                if process.poll() is not None:
                    break
                if ledger_path.exists():
                    try:
                        with sqlite3.connect(str(ledger_path), timeout=0.2) as conn:
                            intents = conn.execute(
                                "SELECT COUNT(*) FROM attempt_intents"
                            ).fetchone()[0]
                    except sqlite3.Error:
                        intents = 0
                    if intents > 0:
                        break
            self.assertIsNone(process.poll(), "child completed before forced termination")
            self.assertGreater(intents, 0, "no in-flight intent was observed before kill")
            process.kill()
            process.wait(timeout=5)
            process.communicate(timeout=5)
            self.assertLess(process.returncode, 0)
            time.sleep(0.8)

            recovery = run_cli(
                "recover",
                "--spec",
                str(spec_path),
                "--ledger",
                str(ledger_path),
                "--snapshot-dir",
                str(snapshot_dir),
                "--log-file",
                str(root / "recover.jsonl"),
            )
            self.assertEqual(recovery.returncode, 0, recovery.stderr)
            payload = json.loads(recovery.stdout)
            self.assertEqual(payload["status"], "recovered_and_rerun")
            self.assertFalse(payload["exit_report_available"])
            self.assertIsNone(payload["pre_crash_exit_report"])
            self.assertGreaterEqual(len(payload["recovery"]["unknown_attempts"]), 1)
            self.assertTrue(payload["rerun"]["results"])
            self.assertEqual(
                payload["recovery"]["state_source"], "committed_ledger_attempts_only"
            )
            with sqlite3.connect(str(ledger_path)) as conn:
                incomplete = conn.execute(
                    "SELECT COUNT(*) FROM requests WHERE status!='complete'"
                ).fetchone()[0]
                unknown = conn.execute("SELECT COUNT(*) FROM unknown_attempts").fetchone()[0]
            self.assertEqual(incomplete, 0)
            self.assertGreaterEqual(unknown, 1)


if __name__ == "__main__":
    unittest.main()
