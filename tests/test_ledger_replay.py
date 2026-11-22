from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from txdiag.canonical import canonical_text, digest
from txdiag.engine import execute
from txdiag.errors import ReplayError
from txdiag.ledger import Ledger, ReplayHooks
from txdiag.model import RuntimeConfig
from txdiag.spec import load_spec

from .helpers import ROOT, load_json, run_cli, sample


class LedgerReplayTests(unittest.TestCase):
    def test_round_ledger_snapshot_and_replay(self) -> None:
        spec = load_spec(sample("all_isolation"))
        cfg = RuntimeConfig(max_running=8, wait_timeout_ms=1000, wait_budget_ms=5000)
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.sqlite"
            snapshot_path = Path(tmp) / "session.snapshot.json"
            ledger = Ledger(ledger_path, snapshot_path, spec, cfg, "row")
            opened = ledger.open()
            self.assertFalse(opened["replaced"])
            result = execute(spec, cfg, "row", hooks=ledger)
            ledger.complete(result)
            info = ledger.session_info()
            self.assertTrue(info["complete"])
            self.assertEqual(info["round_count"], len(result.rounds))
            ledger.close()

            conn = sqlite3.connect(ledger_path)
            rounds = conn.execute(
                "SELECT round_no,format_version,payload_json,content_sha256 FROM rounds "
                "ORDER BY round_no").fetchall()
            events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            conn.close()
            self.assertEqual(len(rounds), len(result.rounds))
            for round_no, version, payload_json, content_sha in rounds:
                self.assertEqual(version, 1)
                self.assertEqual(content_sha, digest(json.loads(payload_json)))
            self.assertGreater(events, 0)
            snapshot = load_json(snapshot_path)
            self.assertEqual(snapshot["format_version"], 1)
            self.assertEqual(snapshot["state"]["phase"], "complete")

            ledger = Ledger(ledger_path, snapshot_path, spec, cfg, "row")
            opened = ledger.open()
            self.assertFalse(opened["replaced"])
            hooks = ReplayHooks(ledger.expected_rounds, complete=True)
            replay = execute(spec, cfg, "row", hooks=hooks)
            self.assertTrue(hooks.summary()["prefix_equal"])
            self.assertEqual(replay.status, result.status)
            ledger.close()

    def test_snapshot_replacement_lists_stable_diffs(self) -> None:
        spec = load_spec(sample("wait"))
        cfg = RuntimeConfig(max_running=8, wait_timeout_ms=1000, wait_budget_ms=5000)
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.sqlite"
            snapshot_path = Path(tmp) / "session.snapshot.json"
            ledger = Ledger(ledger_path, snapshot_path, spec, cfg, "row")
            ledger.open()
            result = execute(spec, cfg, "row", hooks=ledger)
            ledger.complete(result)
            ledger.close()

            changed = RuntimeConfig(max_running=3, wait_timeout_ms=1000,
                                    wait_budget_ms=5000)
            ledger = Ledger(ledger_path, snapshot_path, spec, changed, "row")
            opened = ledger.open()
            self.assertTrue(opened["replaced"])
            self.assertEqual(opened["committed_rounds"], 0)
            diffs = opened["replacement"]["stable_differences"]
            paths = [item["path"] for item in diffs]
            self.assertEqual(paths, sorted(paths))
            self.assertIn("$.ledger.session_digest", paths)
            self.assertEqual(opened["replacement"]["action"], "whole_snapshot_replaced")
            ledger.close()

    def test_replay_tamper_reports_first_event(self) -> None:
        spec = load_spec(sample("wait"))
        cfg = RuntimeConfig(max_running=4, wait_timeout_ms=1000, wait_budget_ms=5000)
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.sqlite"
            snapshot_path = Path(tmp) / "session.snapshot.json"
            ledger = Ledger(ledger_path, snapshot_path, spec, cfg, "row")
            ledger.open()
            result = execute(spec, cfg, "row", hooks=ledger)
            ledger.complete(result)
            hooks = ReplayHooks(ledger.expected_rounds, complete=True, tamper=True)
            with self.assertRaises(ReplayError) as ctx:
                execute(spec, cfg, "row", hooks=hooks)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(ctx.exception.txid, "HOLDER")
            self.assertEqual(ctx.exception.row, "lock:A")
            self.assertTrue(ctx.exception.field)
            ledger.close()

    def test_replay_compares_final_versions_and_isolation_state(self) -> None:
        spec = load_spec(sample("all_isolation"))
        cfg = RuntimeConfig(max_running=8, wait_timeout_ms=1000, wait_budget_ms=5000)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.sqlite", Path(tmp) / "snap.json",
                            spec, cfg, "row")
            ledger.open()
            result = execute(spec, cfg, "row", hooks=ledger)
            ledger.complete(result)
            expected_final = json.loads(canonical_text(result.result))
            expected_final["final_versions"]["acct:A"][0]["value"] = 999
            hooks = ReplayHooks(ledger.expected_rounds, complete=True,
                                expected_final=expected_final)
            with self.assertRaises(ReplayError) as ctx:
                execute(spec, cfg, "row", hooks=hooks)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(ctx.exception.txid, "RC1")
            self.assertEqual(ctx.exception.row, "acct:A")
            self.assertEqual(ctx.exception.field, "value")
            ledger.close()

    def test_forced_termination_recovery_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "crash.sqlite"
            snapshot = Path(tmp) / "crash.snapshot.json"
            report = Path(tmp) / "crash-report.json"
            report.write_text('{"stale":true}\n', encoding="utf-8")
            crashed = run_cli(
                "run", "--spec", str(sample("wait")), "--ledger", str(ledger),
                "--snapshot", str(snapshot), "--inject", "crash-after-round=2",
                "--report", str(report), "--quiet")
            self.assertEqual(crashed.returncode, 70, crashed.stderr)
            self.assertFalse(report.exists(), "forced termination must not emit an exit report")
            conn = sqlite3.connect(ledger)
            committed = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
            conn.close()
            self.assertEqual(committed, 2)

            recovered = run_cli(
                "recover", "--spec", str(sample("wait")), "--ledger", str(ledger),
                "--snapshot", str(snapshot), "--report", str(Path(tmp) / "recover.json"),
                "--quiet")
            self.assertEqual(recovered.returncode, 6)
            recover_report = load_json(Path(tmp) / "recover.json")
            self.assertEqual(recover_report["status"], "INCOMPLETE_REQUIRES_RERUN")
            self.assertFalse(recover_report["exit_report_available"])
            self.assertEqual(recover_report["session"]["unknown_txids"], ["HOLDER", "WAITER"])

            replay_before = run_cli(
                "replay", "--spec", str(sample("wait")), "--ledger", str(ledger),
                "--snapshot", str(snapshot), "--report", str(Path(tmp) / "before.json"),
                "--quiet")
            self.assertEqual(replay_before.returncode, 6)
            before = load_json(Path(tmp) / "before.json")
            self.assertEqual(before["status"], "INCOMPLETE_REQUIRES_RERUN")
            self.assertFalse(before["exit_report_available"])

            resumed = run_cli(
                "run", "--spec", str(sample("wait")), "--ledger", str(ledger),
                "--snapshot", str(snapshot), "--report", str(Path(tmp) / "resume.json"),
                "--quiet")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            resume_report = load_json(Path(tmp) / "resume.json")
            self.assertTrue(resume_report["session"]["complete"])

            replay_after = run_cli(
                "replay", "--spec", str(sample("wait")), "--ledger", str(ledger),
                "--snapshot", str(snapshot), "--report", str(Path(tmp) / "after.json"),
                "--quiet")
            self.assertEqual(replay_after.returncode, 0, replay_after.stderr)
            after = load_json(Path(tmp) / "after.json")
            self.assertEqual(after["status"], "REPLAY_OK")
            self.assertTrue(after["replay"]["prefix_equal"])

    def test_mid_round_termination_marks_running_transactions_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "mid-crash.sqlite"
            snapshot = Path(tmp) / "mid-crash.snapshot.json"
            crashed = run_cli(
                "run", "--spec", str(sample("wait")), "--ledger", str(ledger),
                "--snapshot", str(snapshot), "--inject", "crash-in-round=2",
                "--quiet")
            self.assertEqual(crashed.returncode, 70)
            conn = sqlite3.connect(ledger)
            committed = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
            inflight = conn.execute(
                "SELECT round_no,running_json FROM inflight").fetchall()
            conn.close()
            self.assertEqual(committed, 1)
            self.assertEqual(len(inflight), 1)
            self.assertEqual(json.loads(inflight[0][1]), ["HOLDER", "WAITER"])

            recovered = run_cli(
                "recover", "--spec", str(sample("wait")), "--ledger", str(ledger),
                "--snapshot", str(snapshot), "--report", str(Path(tmp) / "report.json"),
                "--quiet")
            self.assertEqual(recovered.returncode, 6)
            report = load_json(Path(tmp) / "report.json")
            self.assertEqual(report["session"]["unknown_txids"], ["HOLDER", "WAITER"])
            self.assertTrue(report["ledger"]["recovery"]["had_inflight"])
            self.assertFalse(report["exit_report_available"])


if __name__ == "__main__":
    unittest.main()
