from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from config_ledger.errors import ReplayError, SnapshotError
from config_ledger.service import DiagnosticService

from .helpers import (
    log_records,
    make_spec_data,
    running_servers,
    temp_workspace,
    write_json,
)


ROOT = Path(__file__).resolve().parent.parent


def _scenario() -> dict:
    return {
        "node-a": {
            "observations": {
                "alpha": [{"value": "x", "observation_seq": 1, "timestamp_ms": 1000}],
                "numeric": [{"value": 10, "observation_seq": 2, "timestamp_ms": 2000}],
            }
        },
        "node-b": {
            "observations": {
                "alpha": [{"value": "x", "observation_seq": 2, "timestamp_ms": 1100}],
                "numeric": [{"value": 10, "observation_seq": 3, "timestamp_ms": 2100}],
            }
        },
        "node-c": {
            "observations": {
                "alpha": [{"value": "x", "observation_seq": 3, "timestamp_ms": 1200}],
                "numeric": [{"value": 10, "observation_seq": 4, "timestamp_ms": 2200}],
            }
        },
    }


KEYS = {
    "alpha": {"expected": "x", "quorum_bp": 10000},
    "numeric": {"expected": 10, "quorum_bp": 6667},
}


class LedgerAndServiceTests(unittest.TestCase):
    def test_committed_snapshot_restores_and_replays(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            spec_path = write_json(
                temp / "spec.json", make_spec_data(ports, KEYS, timeout_ms=500)
            )
            ledger_path = temp / "ledger.sqlite"
            log_path = temp / "structured.jsonl"
            service = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            report = service.run_round()
            digest = report["content_digest"]
            round_id = report["round_id"]
            service.close()

            reopened = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            try:
                self.assertTrue(reopened.snapshot["valid"])
                self.assertEqual(reopened.snapshot["content_digest"], digest)
                view = reopened.query_dashboard()
                self.assertEqual(view["meta"]["current_round_id"], round_id)
                self.assertEqual(view["meta"]["current_content_digest"], digest)
                replay = reopened.replay()
                self.assertEqual(replay["status"], "MATCH")
                self.assertEqual(replay["round_id"], round_id)
                self.assertEqual(len(replay["divergence_keys"]), 0)
            finally:
                reopened.close()

    def test_parameter_change_stales_old_view_and_records_whole_replacement(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            original = make_spec_data(ports, KEYS, timeout_ms=500)
            spec_path = write_json(temp / "spec.json", original)
            ledger_path = temp / "ledger.sqlite"
            log_path = temp / "structured.jsonl"
            service = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            try:
                first = service.run_round()
                old_round_id = first["round_id"]
                changed = json.loads(json.dumps(original, ensure_ascii=False))
                changed["keys"]["alpha"]["expected"] = "y"
                write_json(spec_path, changed)
                activation = service.reload_spec(spec_path)
                self.assertTrue(activation.changed)
                self.assertEqual(activation.previous_version, 1)
                self.assertEqual(activation.current_version, 2)

                stale_view = service.query_dashboard()
                self.assertEqual(stale_view["meta"]["parameter_version"], 2)
                self.assertIsNone(stale_view["round"])
                self.assertEqual(stale_view["conclusions"], [])
                self.assertEqual(stale_view["fetch_latency"], [])

                connection = sqlite3.connect(ledger_path)
                try:
                    stale_row = connection.execute(
                        "SELECT is_stale, stale_reason FROM rounds WHERE round_id=?",
                        (old_round_id,),
                    ).fetchone()
                finally:
                    connection.close()
                self.assertEqual(stale_row[0], 1)
                self.assertEqual(stale_row[1], "PARAMETER_CHANGED")

                second = service.run_round()
                self.assertEqual(second["parameter_version"], 2)
                view = service.query_dashboard()
                self.assertEqual(len(view["conclusions"]), 2)
                events = [e for e in view["replacement_events"] if e["to_version"] == 2]
                self.assertTrue(events)
                affected = {
                    item["key"]: item for item in events[0]["detail"]["affected_keys"]
                }
                self.assertIn("alpha", affected)
                self.assertEqual(affected["alpha"]["old"]["status"], "OK")
                self.assertEqual(
                    affected["alpha"]["new"]["status"], "EXPECTED_MISMATCH"
                )
                self.assertTrue(affected["alpha"]["decision_flip"])
                self.assertTrue(affected["alpha"]["parameter_caused"])
                self.assertEqual(events[0]["detail"]["replacement_semantics"], "whole-current-view")
            finally:
                service.close()

    def test_ui_query_budget_logs_e_latbudget_without_dropping_samples(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            spec_path = write_json(
                temp / "spec.json",
                make_spec_data(ports, KEYS, timeout_ms=500, latency_budget_ms=0),
            )
            log_path = temp / "structured.jsonl"
            service = DiagnosticService(
                spec_path, ledger_path=temp / "ledger.sqlite", log_path=log_path
            )
            try:
                service.run_round()
                first = service.query_dashboard()
                second = service.query_dashboard()
                self.assertTrue(first["query_latency"]["over_budget"])
                self.assertGreater(first["query_latency"]["overrun_us"], 0)
                self.assertEqual(second["ui_latency"]["sample_count"], 2)
                self.assertEqual(second["ui_latency"]["budget_ms"], 0)
                self.assertTrue(second["ui_latency"]["all_samples_retained"])
                self.assertGreater(second["ui_latency"]["max_us"], 0)
                events = [
                    record
                    for record in log_records(log_path)
                    if record["code"] == "E_LATBUDGET"
                ]
                self.assertEqual(len(events), 2)
                self.assertGreater(events[0]["overrun_us"], 0)
            finally:
                service.close()

    def test_restart_between_parameter_change_and_round_keeps_old_view_for_replacement(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            original = make_spec_data(ports, KEYS, timeout_ms=500)
            spec_path = write_json(temp / "spec.json", original)
            ledger_path = temp / "ledger.sqlite"
            log_path = temp / "structured.jsonl"
            first_service = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            first_service.run_round()
            first_service.close()

            changed = json.loads(json.dumps(original, ensure_ascii=False))
            changed["keys"]["alpha"]["expected"] = "y"
            write_json(spec_path, changed)
            activation_service = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            self.assertEqual(activation_service.spec.parameter_version, 2)
            activation_service.close()

            after_restart = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            try:
                self.assertIsNone(after_restart.query_dashboard()["round"])
                after_restart.run_round()
                view = after_restart.query_dashboard()
                event = next(
                    event
                    for event in view["replacement_events"]
                    if event["to_version"] == 2
                )
                affected = {
                    item["key"]: item for item in event["detail"]["affected_keys"]
                }
                self.assertEqual(affected["alpha"]["old"]["status"], "OK")
                self.assertEqual(
                    affected["alpha"]["new"]["status"], "EXPECTED_MISMATCH"
                )
                self.assertTrue(affected["alpha"]["decision_flip"])
            finally:
                after_restart.close()

    def test_snapshot_digest_mismatch_logs_e_snapshot_and_invalidates_view(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            spec_path = write_json(
                temp / "spec.json", make_spec_data(ports, KEYS, timeout_ms=500)
            )
            ledger_path = temp / "ledger.sqlite"
            log_path = temp / "structured.jsonl"
            service = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            service.run_round()
            service.close()
            connection = sqlite3.connect(ledger_path)
            try:
                with connection:
                    connection.execute(
                        "UPDATE meta SET value='tampered-digest' "
                        "WHERE key='current_content_digest'"
                    )
            finally:
                connection.close()
            with self.assertRaises(SnapshotError):
                DiagnosticService(spec_path, ledger_path=ledger_path, log_path=log_path)
            codes = {record["code"] for record in log_records(log_path)}
            self.assertIn("E_SNAPSHOT", codes)
            connection = sqlite3.connect(ledger_path)
            try:
                meta = {
                    row[0]: row[1]
                    for row in connection.execute(
                        "SELECT key, value FROM meta WHERE key IN "
                        "('current_round_id','current_content_digest')"
                    )
                }
                cursor = connection.execute(
                    "SELECT phase FROM cursors ORDER BY parameter_version DESC LIMIT 1"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(meta["current_round_id"], "")
            self.assertEqual(meta["current_content_digest"], "INVALID")
            self.assertEqual(cursor[0], "RERUN_REQUIRED")

    def test_replay_reports_first_key_node_field_and_logs_e_replay(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            spec_path = write_json(
                temp / "spec.json", make_spec_data(ports, KEYS, timeout_ms=500)
            )
            ledger_path = temp / "ledger.sqlite"
            log_path = temp / "structured.jsonl"
            service = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            try:
                service.run_round()
                connection = sqlite3.connect(ledger_path)
                try:
                    row = connection.execute(
                        "SELECT decision_json FROM conclusions WHERE key='alpha'"
                    ).fetchone()
                    decision = json.loads(row[0])
                    decision["status"] = "TAMPERED"
                    with connection:
                        connection.execute(
                            "UPDATE conclusions SET decision_json=? WHERE key='alpha'",
                            (json.dumps(decision, ensure_ascii=False, separators=(",", ":")),),
                        )
                finally:
                    connection.close()
                with self.assertRaises(ReplayError) as caught:
                    service.replay()
                diff = caught.exception.first_diff
                self.assertEqual(diff["key"], "alpha")
                self.assertEqual(diff["node_id"], "*")
                self.assertEqual(diff["field"], "status")
                self.assertEqual(diff["ledger"], "TAMPERED")
                self.assertEqual(diff["replay"], "OK")
                self.assertIn("E_REPLAY", {r["code"] for r in log_records(log_path)})
            finally:
                service.close()

    def test_replay_detects_tampered_latency_aggregate(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            spec_path = write_json(
                temp / "spec.json", make_spec_data(ports, KEYS, timeout_ms=500)
            )
            ledger_path = temp / "ledger.sqlite"
            log_path = temp / "structured.jsonl"
            service = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            try:
                service.run_round()
                connection = sqlite3.connect(ledger_path)
                try:
                    row = connection.execute(
                        "SELECT stats_json FROM latency_aggregates "
                        "WHERE scope='overall' LIMIT 1"
                    ).fetchone()
                    stats = json.loads(row[0])
                    stats["mean_us"] = int(stats["mean_us"]) + 1
                    with connection:
                        connection.execute(
                            "UPDATE latency_aggregates SET stats_json=? "
                            "WHERE scope='overall'",
                            (json.dumps(stats, ensure_ascii=False, separators=(",", ":")),),
                        )
                finally:
                    connection.close()
                with self.assertRaises(ReplayError) as caught:
                    service.replay()
                diff = caught.exception.first_diff
                self.assertEqual(diff["key"], "<latency>")
                self.assertEqual(diff["node_id"], "*")
                self.assertEqual(diff["field"], "mean_us")
            finally:
                service.close()

    def test_forced_termination_leaves_only_unknown_attempt_and_no_exit_report(self) -> None:
        with temp_workspace() as temp, running_servers(_scenario()) as ports:
            spec_path = write_json(
                temp / "spec.json", make_spec_data(ports, KEYS, timeout_ms=500)
            )
            ledger_path = temp / "ledger.sqlite"
            log_path = temp / "structured.jsonl"
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tools" / "crash_worker.py"),
                    str(spec_path),
                    str(ledger_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 10
                found = False
                while time.monotonic() < deadline:
                    try:
                        connection = sqlite3.connect(ledger_path, timeout=0.2)
                        try:
                            row = connection.execute(
                                "SELECT status FROM attempts ORDER BY started_at DESC LIMIT 1"
                            ).fetchone()
                        finally:
                            connection.close()
                        if row and row[0] == "IN_PROGRESS":
                            found = True
                            break
                    except sqlite3.Error:
                        pass
                    if child.poll() is not None:
                        break
                    time.sleep(0.05)
                self.assertTrue(found, "child did not commit an IN_PROGRESS attempt")
            finally:
                child.kill()
                child.wait(timeout=5)

            reopened = DiagnosticService(
                spec_path, ledger_path=ledger_path, log_path=log_path
            )
            try:
                self.assertEqual(len(reopened.recovery.unknown_attempts), 1)
                view = reopened.query_dashboard()
                self.assertIsNone(view["round"])
                self.assertEqual(len(view["unknown_attempts"]), 1)
                self.assertEqual(view["unknown_attempts"][0]["status"], "UNKNOWN")
                self.assertFalse(view["unknown_attempts"][0].get("exit_report"))
                self.assertIn("E_RECOVERY", {r["code"] for r in log_records(log_path)})
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
