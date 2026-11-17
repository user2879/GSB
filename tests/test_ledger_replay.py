from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autoscale_diag.engine import Simulation, simulate
from autoscale_diag.errors import BusinessError, E_LEDGER, E_REPLAY
from autoscale_diag.ledger import Ledger
from autoscale_diag.replay import replay_ledger
from autoscale_diag.reporting import build_report
from autoscale_diag.spec import load_spec

from tests.helpers import SPEC_V1, SPEC_V2


class LedgerReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "ledger.db"
        self.spec = load_spec(SPEC_V1)
        self.result = simulate(self.spec)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def create_complete(self) -> None:
        with Ledger(self.db) as ledger:
            ledger.create_run(self.spec, run_id="run", path=self.result.path)
            for minute in self.result.minutes:
                ledger.commit_minute("run", minute, self.spec.minute_count)
            ledger.finalize_run("run", self.spec.minute_count)

    def test_complete_ledger_replays_cleanly(self) -> None:
        self.create_complete()
        with Ledger(self.db) as ledger:
            replay = replay_ledger(ledger, self.spec, "run")
            report = build_report(ledger, "run")
        self.assertEqual(replay.verified_minutes, 12)
        self.assertEqual(replay.verified_decisions, 36)
        self.assertTrue(report["exit_report_available"])
        self.assertEqual(report["run"]["status"], "complete")
        self.assertEqual(report["summary"]["cooling_suppressions"], 8)

    def test_prefix_recovery_marks_unknown_minute_and_continues_identically(self) -> None:
        with Ledger(self.db) as ledger:
            ledger.create_run(self.spec, run_id="run", path=self.result.path)
            for minute in self.result.minutes[:4]:
                ledger.commit_minute("run", minute, self.spec.minute_count)
            self.assertEqual(ledger.committed_count("run"), 4)
        with Ledger(self.db) as ledger:
            run = ledger.prepare_resume("run", self.spec.minute_count)
            self.assertEqual(run["status"], "interrupted")
            events = ledger.recovery_events("run")
            self.assertEqual(events[0]["event_type"], "UNKNOWN_MINUTE_REQUIRES_RERUN")
            self.assertEqual(events[0]["minute_index"], 4)
            replay = replay_ledger(ledger, self.spec, "run", limit=4)
            self.assertEqual(replay.verified_minutes, 4)
            ledger.mark_running("run")
            states = ledger.latest_states("run", [item.id for item in self.spec.workloads])
            simulation = Simulation(self.spec, requested_path=run["path"])
            simulation.states = states
            simulation.prime(4)
            for expected in self.result.minutes[4:]:
                actual = simulation.step()
                self.assertEqual(actual.canonical_payload(), expected.canonical_payload())
                ledger.commit_minute("run", actual, self.spec.minute_count)
            ledger.finalize_run("run", self.spec.minute_count)
            report = build_report(ledger, "run")
        self.assertTrue(report["exit_report_available"])
        self.assertEqual(report["ledger_digest"], build_report_digest(self.db, self.spec))

    def test_all_rows_but_missing_finalize_are_not_an_exit_report(self) -> None:
        with Ledger(self.db) as ledger:
            ledger.create_run(self.spec, run_id="run", path=self.result.path)
            for minute in self.result.minutes:
                ledger.commit_minute("run", minute, self.spec.minute_count)
            report = build_report(ledger, "run")
            self.assertFalse(report["exit_report_available"])
            self.assertIn("E_LEDGER", report["business_codes"])
        with Ledger(self.db) as ledger:
            ledger.prepare_resume("run", self.spec.minute_count)
            event_types = [item["event_type"] for item in ledger.recovery_events("run")]
            self.assertIn("COMPLETION_UNCONFIRMED", event_types)

    def test_declaration_version_mismatch_is_ereplay(self) -> None:
        self.create_complete()
        v2 = load_spec(SPEC_V2)
        with Ledger(self.db) as ledger, self.assertRaises(BusinessError) as context:
            replay_ledger(ledger, v2, "run")
        self.assertEqual(context.exception.code, E_REPLAY)
        self.assertEqual(context.exception.field_path, "declaration_version")
        self.assertIn("basis", context.exception.details)

    def test_same_version_budget_mismatch_reports_parameter_field(self) -> None:
        self.create_complete()
        changed = replace(self.spec, global_replica_budget=7)
        with Ledger(self.db) as ledger, self.assertRaises(BusinessError) as context:
            replay_ledger(ledger, changed, "run")
        self.assertEqual(context.exception.code, E_REPLAY)
        self.assertEqual(context.exception.field_path, "global_replica_budget")
        self.assertEqual(context.exception.details["ledger_value"], 6)
        self.assertEqual(context.exception.details["current_value"], 7)

    def test_first_replay_difference_is_located(self) -> None:
        self.create_complete()
        with Ledger(self.db) as ledger:
            with ledger.conn:
                ledger.conn.execute(
                    """
                    UPDATE workload_decisions SET expected_replicas = expected_replicas + 1
                    WHERE run_id='run' AND minute_index=2 AND workload_id='checkout'
                    """
                )
            with self.assertRaises(BusinessError) as context:
                replay_ledger(ledger, self.spec, "run")
        details = context.exception.details
        self.assertEqual(context.exception.code, E_REPLAY)
        self.assertEqual(context.exception.field_path, "expected_replicas")
        self.assertEqual(details["minute_index"], 2)
        self.assertEqual(details["workload_id"], "checkout")

    def test_observation_corruption_is_located(self) -> None:
        self.create_complete()
        with Ledger(self.db) as ledger:
            with ledger.conn:
                ledger.conn.execute(
                    """
                    UPDATE workload_decisions SET observation_json = ?
                    WHERE run_id='run' AND minute_index=1 AND workload_id='search'
                    """,
                    ('{"minute":1,"requests":999,"avg_latency_ms":1,"errors":0}',),
                )
            with self.assertRaises(BusinessError) as context:
                replay_ledger(ledger, self.spec, "run")
        self.assertEqual(context.exception.field_path, "observation")
        self.assertEqual(context.exception.details["minute_index"], 1)
        self.assertEqual(context.exception.details["workload_id"], "search")

    def test_failed_minute_transaction_rolls_back_all_rows(self) -> None:
        with Ledger(self.db) as ledger:
            ledger.create_run(self.spec, run_id="run", path=self.result.path)
            ledger.commit_minute("run", self.result.minutes[0], self.spec.minute_count)
            ledger.conn.execute(
                """
                CREATE TRIGGER force_cooling_failure
                BEFORE INSERT ON cooling_suppressions
                WHEN NEW.workload_id = 'payments'
                BEGIN
                  SELECT RAISE(ABORT, 'forced cooling persistence failure');
                END;
                """
            )
            ledger.conn.commit()
            with self.assertRaises(BusinessError) as context:
                ledger.commit_minute("run", self.result.minutes[1], self.spec.minute_count)
            self.assertEqual(context.exception.code, E_LEDGER)
            self.assertEqual(ledger.committed_count("run"), 1)
            rows = ledger.conn.execute(
                "SELECT COUNT(*) FROM workload_decisions WHERE minute_index=1"
            ).fetchone()[0]
            self.assertEqual(rows, 0)

    def test_replace_clears_old_recovery_events(self) -> None:
        with Ledger(self.db) as ledger:
            ledger.create_run(self.spec, run_id="run", path=self.result.path)
            ledger.commit_minute("run", self.result.minutes[0], self.spec.minute_count)
            ledger.prepare_resume("run", self.spec.minute_count)
            self.assertTrue(ledger.recovery_events("run"))
            replaced = load_spec(SPEC_V2)
            ledger.create_run(
                replaced,
                run_id="run",
                path="incremental",
                replace=True,
            )
            self.assertEqual(ledger.recovery_events("run"), [])
            self.assertEqual(ledger.get_run("run")["declaration_version"], "demo-v2")


def build_report_digest(db: Path, spec) -> str:
    with Ledger(db) as ledger:
        return build_report(ledger, "run")["ledger_digest"]


if __name__ == "__main__":
    unittest.main()
