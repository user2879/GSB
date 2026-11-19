from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from streamsim.samples import generate_samples
from tests.helpers import (
    ROOT,
    make_session_fixture,
    read_json,
    run_cli,
    session_dirs,
    write_json,
)


class SimulationTests(unittest.TestCase):
    def test_continuous_simulation_and_log_correspondence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate_samples(root)
            code, report = run_cli(root / "session.json")
            self.assertEqual(code, 0)
            self.assertTrue(report["summary"]["finished"])
            self.assertEqual(report["summary"]["fragment_downloads_completed"], 30)
            self.assertEqual(report["summary"]["decision_count"], 31)
            self.assertGreaterEqual(report["summary"]["switch_count"], 1)
            state_dir, _ = session_dirs(root / "session.json")
            journal = [
                json.loads(line)
                for line in (state_dir / "simulation.log.jsonl").read_text().splitlines()
            ]
            self.assertEqual(journal, report["events"])
            self.assertEqual(len(journal), report["summary"]["event_count"])

    def test_concurrency_does_not_change_decision_or_buffer_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate_samples(root)
            code2, report2 = run_cli(root / "session.json")
            code1, report1 = run_cli(root / "session_concurrency1.json")
            self.assertEqual(code1, 0)
            self.assertEqual(code2, 0)
            self.assertEqual(report1["events"], report2["events"])
            self.assertEqual(report1["decisions"], report2["decisions"])
            self.assertEqual(report1["summary"], report2["summary"])
            self.assertEqual(report2["prefetch"]["peak_active_tasks"], 2)
            self.assertEqual(report1["prefetch"]["peak_active_tasks"], 1)
            self.assertLessEqual(
                report2["prefetch"]["peak_active_bytes"],
                report2["prefetch"]["memory_budget_bytes"],
            )

    def test_scripted_termination_resume_matches_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate_samples(root)
            code_base, baseline = run_cli(root / "session.json")
            self.assertEqual(code_base, 0)

            code_int, interrupted = run_cli(root / "session_interrupt.json")
            self.assertEqual(code_int, 75)
            self.assertTrue(interrupted["interrupted"])
            state_dir, _ = session_dirs(root / "session_interrupt.json")
            self.assertTrue((state_dir / "interrupted.json").exists())

            code_resume, resumed = run_cli(root / "session_interrupt.json")
            self.assertEqual(code_resume, 0)
            self.assertEqual(resumed["recovery_start"]["mode"], "resume")
            self.assertTrue(resumed["recovery_start"]["had_interruption_marker"])
            self.assertEqual(resumed["events"], baseline["events"])
            self.assertEqual(resumed["decisions"], baseline["decisions"])
            self.assertEqual(resumed["summary"], baseline["summary"])
            log_journal = [
                json.loads(line)
                for line in (state_dir / "simulation.log.jsonl").read_text().splitlines()
            ]
            self.assertEqual(log_journal, resumed["events"])

    def test_tampered_snapshot_returns_e_digest_then_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate_samples(root)
            code_base, baseline = run_cli(root / "session.json")
            self.assertEqual(code_base, 0)
            code_int, _ = run_cli(root / "session_interrupt.json")
            self.assertEqual(code_int, 75)

            state_dir, _ = session_dirs(root / "session_interrupt.json")
            current = state_dir / "snapshot.json"
            data = read_json(current)
            data["state"]["buffer_ms"] = int(data["state"]["buffer_ms"]) + 17
            write_json(current, data)

            verify_session = root / "session_interrupt.json"
            verify_code, verify_payload = run_cli_verify(verify_session)
            self.assertEqual(verify_code, 4)
            self.assertEqual(verify_payload["error"]["code"], "E_DIGEST")

            code_resume, resumed = run_cli(verify_session)
            self.assertEqual(code_resume, 0)
            self.assertEqual(resumed["recovery_start"]["mode"], "archive_fallback")
            self.assertEqual(
                resumed["recovery_start"]["invalid_snapshots"][0]["code"], "E_DIGEST"
            )
            self.assertEqual(resumed["integrity"]["code"], "E_DIGEST")
            self.assertEqual(resumed["events"], baseline["events"])
            self.assertEqual(resumed["summary"], baseline["summary"])
            _, snapshot_dir = session_dirs(verify_session)
            quarantine = list((snapshot_dir / "quarantine").rglob("snapshot.json-*"))
            self.assertTrue(quarantine)

    def test_version_change_archives_old_state_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate_samples(root)
            code_old, old_report = run_cli(root / "session.json")
            self.assertEqual(code_old, 0)
            session_payload = read_json(root / "session.json")
            changed = dict(session_payload)
            changed_client = dict(session_payload["client"])
            changed_client["max_buffer_ms"] = 11_000
            changed["client"] = changed_client
            version2 = root / "session_version2.json"
            write_json(version2, changed)
            code_new, new_report = run_cli(version2)
            self.assertEqual(code_new, 0)
            self.assertEqual(new_report["recovery_start"]["mode"], "version_reset")
            self.assertNotEqual(new_report["version_id"], old_report["version_id"])
            self.assertTrue(new_report["recovery_start"]["archived_paths"])
            _, snapshot_dir = session_dirs(version2)
            stale_files = list((snapshot_dir / "stale" / "version_changed").iterdir())
            self.assertTrue(stale_files)

    def test_sigterm_only_stops_at_event_boundary_and_resumes_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generate_samples(root)
            code_base, baseline = run_cli(root / "session.json")
            self.assertEqual(code_base, 0)
            signal_session = write_signal_session(root)
            command = [
                sys.executable,
                "-m",
                "streamsim",
                "run",
                str(signal_session),
                "--event-delay-ms",
                "15",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            state_dir, _ = session_dirs(signal_session)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not (state_dir / "run_meta.json").exists():
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            self.assertIsNone(process.poll(), "process exited before SIGTERM was sent")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 75, stderr)
            payload = json.loads(stdout.strip())
            self.assertTrue(payload["interrupted"])
            self.assertEqual(payload["marker"]["signal"], "SIGTERM")
            self.assertTrue((state_dir / "interrupted.json").exists())

            code_resume, resumed = run_cli(signal_session)
            self.assertEqual(code_resume, 0)
            self.assertEqual(resumed["events"], baseline["events"])
            self.assertEqual(resumed["summary"], baseline["summary"])
            self.assertTrue(resumed["recovery_start"]["had_interruption_marker"])

    def test_buffer_empty_is_counted_on_slow_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fragments = []
            for _ in range(6):
                fragments.append(
                    {
                        "duration_ms": 2000,
                        "representations": [
                            {"bitrate_bps": 200_000, "size_bytes": 50_000},
                            {"bitrate_bps": 400_000, "size_bytes": 100_000},
                        ],
                    }
                )
            session = make_session_fixture(
                root,
                fragments=fragments,
                trace=[
                    {"bandwidth_bps": 1_500_000, "rtt_ms": 10},
                    {"bandwidth_bps": 90_000, "rtt_ms": 10},
                ],
                events=[{"at_ms": 5000, "action": "trace_switch", "segment_index": 1}],
                client={
                    "startup_buffer_ms": 1000,
                    "max_buffer_ms": 4000,
                    "safety_margin_ms": 0,
                    "throughput_safety_ratio": 1.0,
                    "concurrency": 1,
                },
            )
            code, report = run_cli(session)
            self.assertEqual(code, 0)
            self.assertGreaterEqual(report["summary"]["drain_count"], 1)
            empty_events = [event for event in report["events"] if event["kind"] == "buffer_empty"]
            self.assertTrue(empty_events)
            self.assertTrue(all(event["buffer_ms"] == 0 for event in empty_events))


def run_cli_verify(session_path: Path) -> tuple[int, dict]:
    process = subprocess.run(
        [sys.executable, "-m", "streamsim", "verify-snapshots", str(session_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode, json.loads(process.stdout)


def write_signal_session(root: Path) -> Path:
    session = read_json(root / "session.json")
    session["state_dir"] = "sample_runs/sigterm/state"
    session["snapshot_dir"] = "sample_runs/sigterm/snapshots"
    path = root / "session_sigterm.json"
    write_json(path, session)
    return path


if __name__ == "__main__":
    unittest.main()
