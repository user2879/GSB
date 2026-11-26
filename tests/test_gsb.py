from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
import os
import signal
import subprocess
import sys
from pathlib import Path

from gsb.acceptance import json_equal, run_acceptance
from gsb.errors import SchemaViolation, SnapshotUnavailable, SnapshotVersionMismatch
from gsb.incremental import IncrementalEngine
from gsb.pipeline import build_work_items, run_corpus
from gsb.samplegen import generate_sample_dir
from gsb.scanner import scan_corpus
from gsb.scheduler import BudgetScheduler, plan_batch
from gsb.schema import load_sample, validate_bundle
from gsb.settlement import settle_scope
from gsb.snapshot import SnapshotStore


ROOT = Path(__file__).resolve().parents[1]


class SampleTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sample_dir = generate_sample_dir(Path(cls.tmp.name) / "samples")
        cls.bundle = load_sample(cls.sample_dir)
        cls.serial = run_corpus(cls.bundle, use_scheduler=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()


class SchemaTests(SampleTestBase):
    def test_valid_sample_loads(self) -> None:
        self.assertEqual(len(self.bundle["tickets"]), 18)
        self.assertIn("app", self.bundle["channels"])

    def test_duplicate_ticket_and_bad_width_report_fields(self) -> None:
        bad = copy.deepcopy(self.bundle)
        bad["tickets"][1]["ticket_id"] = bad["tickets"][0]["ticket_id"]
        bad["config"]["fixed_point"]["decimal_width"] = 3
        with self.assertRaises(SchemaViolation) as ctx:
            validate_bundle(bad)
        self.assertEqual(ctx.exception.code, "E_SCHEMA_VIOLATION")
        fields = {item["field"] for item in ctx.exception.issues}
        self.assertIn("ticket_id", fields)
        self.assertIn("fixed_point.decimal_width", fields)

    def test_unknown_token_identifies_ticket_and_index(self) -> None:
        bad = copy.deepcopy(self.bundle)
        bad["tickets"][0]["tokens"][0] = "NOT_A_VOCAB_TOKEN"
        with self.assertRaises(SchemaViolation) as ctx:
            validate_bundle(bad)
        issue = next(item for item in ctx.exception.issues if item["field"] == "tokens[0]")
        self.assertEqual(issue["id"], bad["tickets"][0]["ticket_id"])


class ScopeTests(SampleTestBase):
    def test_nested_scope_uses_start_then_end_then_trigger_order(self) -> None:
        scan = self.serial["scans"]["TK-0001"]
        self.assertEqual([(scope["start"], scope["end"]) for scope in scan["scopes"]], [(1, 7), (3, 7)])
        self.assertTrue(all(hit["effective_scope_id"] == "TK-0001:S001" for hit in scan["hits"]))
        self.assertEqual(len(scan["suppressions"]), 2)
        self.assertIn("start ASC", scan["suppressions"][0]["basis"])

    def test_overlapping_scopes_are_recorded(self) -> None:
        scan = self.serial["scans"]["TK-0015"]
        self.assertEqual([(scope["start"], scope["end"]) for scope in scan["scopes"]], [(0, 6), (3, 9)])
        suppressed = [item["suppressed_scope_id"] for item in scan["suppressions"]]
        self.assertIn("TK-0015:S003", suppressed)

    def test_scan_order_is_position_order(self) -> None:
        for scan in self.serial["scans"].values():
            positions = [next(hit["position"] for hit in scan["hits"] if hit["hit_id"] == hit_id)
                         for hit_id in scan["scan_order_hit_ids"]]
            self.assertEqual(positions, sorted(positions))


class SettlementCompositionTests(SampleTestBase):
    def test_saturation_steps_and_zeroed_trail_are_retained(self) -> None:
        settlement = self.serial["settlements"]["TK-0005"]
        self.assertGreaterEqual(settlement["saturation_event_count"], 2)
        zeroed = next(hit for hit in settlement["hit_results"] if hit["canceled_to_zero"])
        self.assertTrue(zeroed["cancellation_steps"])
        self.assertTrue(any(step["saturated"] for step in zeroed["cancellation_steps"]))
        self.assertTrue(all(
            step["after_cancellation_q"] >= self.bundle["config"]["score_lower_bound_points"] * settlement["scale"]
            for hit in settlement["hit_results"] for step in hit["cancellation_steps"]
        ))

    def test_boundary_ticket_is_flagged_with_fixed_rule(self) -> None:
        conclusion = self.serial["conclusions_by_id"]["TK-0013"]
        self.assertTrue(conclusion["boundary"])
        self.assertIn("half-step", conclusion["boundary_checks"][0]["decision_rule"])

    def test_float_error_report_has_first_exceeded_stage(self) -> None:
        coarse = run_corpus(self.bundle, decimal_width=2, use_scheduler=False)
        counts = coarse["error_summary"]["first_exceeded_stage_counts"]
        self.assertGreater(counts["cancellation"], 0)
        stages = {error["first_exceeded_stage"] for error in coarse["errors"] if error["first_exceeded_stage"]}
        self.assertTrue(stages <= {"cancellation", "length_penalty", "composition"})


class IncrementalTests(SampleTestBase):
    def test_replace_one_ticket_invalidates_only_it(self) -> None:
        engine = IncrementalEngine(self.bundle)
        ticket = copy.deepcopy(next(t for t in self.bundle["tickets"] if t["ticket_id"] == "TK-0004"))
        ticket["tokens"] = ["客服", "耐心", "处理", "延迟", "。"]
        report = engine.apply({"seq": "test", "op": "replace_ticket", "ticket": ticket})
        self.assertEqual(report["affected_ticket_ids"], ["TK-0004"])
        self.assertEqual(report["reused_ticket_count"], 17)
        fresh = run_corpus(engine.bundle, use_scheduler=False)
        self.assertTrue(json_equal(engine.result["conclusions"], fresh["conclusions"]))

    def test_trigger_order_edit_invalidates_only_referencing_ticket(self) -> None:
        engine = IncrementalEngine(self.bundle)
        report = engine.apply({
            "seq": "test",
            "op": "upsert_neg_trigger",
            "trigger": {"trigger_id": "NEG_BIE", "term": "别", "order": 8},
        })
        self.assertEqual(report["affected_ticket_ids"], ["TK-0003"])
        reasons = {entry["reason"] for entry in report["ledger"]}
        self.assertIn("negation_trigger_referenced", reasons)

    def test_remove_conflict_keeps_cached_scan_consistent(self) -> None:
        engine = IncrementalEngine(self.bundle)
        report = engine.apply({"seq": "test", "op": "remove_conflict", "entry_id": "CONF_DANSHI"})
        self.assertEqual(report["affected_ticket_ids"], ["TK-0001", "TK-0006"])
        fresh = run_corpus(engine.bundle, use_scheduler=False)
        self.assertTrue(json_equal(engine.result["scans"], fresh["scans"]))
        self.assertTrue(json_equal(engine.result["settlements"], fresh["settlements"]))
        self.assertTrue(json_equal(engine.result["conclusions"], fresh["conclusions"]))


class SchedulerTests(SampleTestBase):
    def test_plan_batch_marks_stable_pending_tail(self) -> None:
        items = [
            {"ticket_id": "B", "scope_id": "B:S2", "start": 2, "point_cost": 4},
            {"ticket_id": "A", "scope_id": "A:S0", "start": 0, "point_cost": 5},
            {"ticket_id": "A", "scope_id": "A:S3", "start": 3, "point_cost": 9},
        ]
        eligible, pending, used = plan_batch(items, 6)
        self.assertEqual([item["scope_id"] for item in eligible], ["A:S0"])
        self.assertEqual([item["scope_id"] for item in pending], ["A:S3", "B:S2"])
        self.assertEqual(used, 5)
        self.assertTrue(all(item["error_code"] == "E_SCOPE_BUDGET_PENDING" for item in pending))

    def test_concurrent_results_and_ledger_match_serial(self) -> None:
        for workers in (1, 2, 4):
            result = run_corpus(self.bundle, budget_points=24, max_workers=workers)
            self.assertTrue(result["complete"])
            self.assertTrue(json_equal(result["conclusions"], self.serial["conclusions"]))
            self.assertTrue(json_equal(result["scope_ledger"], self.serial["scope_ledger"]))

    def test_worker_concurrency_never_exceeds_limit(self) -> None:
        active = 0
        peak = 0
        lock = threading.Lock()

        def worker(item: dict) -> dict:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return {"scope_id": item["scope_id"], "ticket_id": item["ticket_id"]}

        items = [{"ticket_id": "T", "scope_id": f"T:S{i}", "start": i, "point_cost": 1} for i in range(8)]
        scheduler = BudgetScheduler(100, 3)
        batch = scheduler.submit_batch(items, worker)
        self.assertEqual(batch["processed_count"], 8)
        self.assertLessEqual(peak, 3)
        self.assertGreaterEqual(batch["peak_in_flight"], 1)

    def test_any_budget_exhaustion_resumes_pending_in_order(self) -> None:
        scans = scan_corpus(self.bundle)
        work_items = build_work_items(scans)
        tickets = {ticket["ticket_id"]: ticket for ticket in self.bundle["tickets"]}
        expected = {
            scope["scope_id"]: scope
            for settlement in self.serial["settlements"].values()
            for scope in settlement["scope_results"]
        }

        def real_worker(item: dict) -> dict:
            scan = scans[item["ticket_id"]]
            scope = next(scope for scope in scan["scopes"] if scope["scope_id"] == item["scope_id"])
            return settle_scope(scan, scope, tickets[item["ticket_id"]], self.bundle)

        for initial_budget in (0, 1, 3, 8, 12, 24):
            for workers in (1, 4):
                scheduler = BudgetScheduler(initial_budget, workers)
                first = scheduler.submit_batch(work_items, real_worker, budget_points=initial_budget)
                results = list(first["results"])
                while scheduler.pending:
                    next_batch = scheduler.submit_batch([], real_worker, budget_points=1000)
                    results.extend(next_batch["results"])
                results.sort(key=lambda result: (result["ticket_id"], result["start"], result["scope_id"]))
                self.assertEqual([item["scope_id"] for item in results], list(expected))
                for result in results:
                    self.assertTrue(json_equal(result, expected[result["scope_id"]]))


class SnapshotTests(SampleTestBase):
    def test_commit_recover_and_corruption_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "snap")
            state = {
                "bundle": self.bundle,
                "global_version": self.serial["global_version"],
                "config_hash": self.serial["config_hash"],
                "scans": self.serial["scans"],
            }
            store.commit_state(state, "hit_scan")
            loaded = store.load_current(expected_config_hash=self.serial["config_hash"])
            self.assertEqual(loaded["manifest"]["stage"], "hit_scan")

            with self.assertRaises(SnapshotVersionMismatch):
                store.load_current(expected_global_version=999)

            manifest_path = store.current_link / "manifest.json"
            manifest_path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(SnapshotUnavailable):
                store.load_current()

    def test_missing_snapshot_does_not_promise_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "missing")
            with self.assertRaises(SnapshotUnavailable):
                store.load_current()

    def test_force_kill_before_commit_recovers_committed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = generate_sample_dir(root / "samples")
            snapshot_dir = root / "snap"
            code = f"""
import os, signal
from pathlib import Path
from gsb.schema import load_sample
from gsb.pipeline import config_hash, scan_corpus
from gsb.snapshot import SnapshotStore
bundle = load_sample({str(sample_dir)!r})
store = SnapshotStore({str(snapshot_dir)!r})
state = {{'bundle': bundle, 'global_version': 1, 'config_hash': config_hash(bundle)}}
store.commit_state(state, 'validation')
state['scans'] = scan_corpus(bundle)
store.commit_state(state, 'hit_scan')
staging = Path({str(snapshot_dir)!r}) / '.staging-killed-partial'
staging.mkdir(parents=True, exist_ok=True)
(staging / '03_scope_settlement.json').write_text('{{partial', encoding='utf-8')
os.kill(os.getpid(), signal.SIGKILL)
"""
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            completed = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, -signal.SIGKILL, completed.stderr)

            bundle = load_sample(sample_dir)
            serial = run_corpus(bundle, use_scheduler=False)
            store = SnapshotStore(snapshot_dir)
            loaded = store.load_current(expected_config_hash=serial["config_hash"])
            self.assertEqual(loaded["manifest"]["stage"], "hit_scan")
            from gsb.pipeline import resume_from_snapshot_state

            resumed = resume_from_snapshot_state(loaded)
            self.assertTrue(json_equal(resumed["conclusions"], serial["conclusions"]))
            self.assertTrue((snapshot_dir / ".staging-killed-partial").exists())


class AcceptanceEndToEndTests(unittest.TestCase):
    def test_full_local_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_acceptance(ROOT / "samples", Path(tmp) / "artifacts")
            self.assertTrue(report["ok"])
            self.assertEqual(len(report["checks"]), 7)


if __name__ == "__main__":
    unittest.main()
