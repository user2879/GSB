from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from scale_review.algorithms import (
    compute_full_minute,
    run_full_path,
    run_incremental_path,
    select_path,
)
from scale_review.allocation import conclusion_bytes
from scale_review.canonical import canonical_bytes
from scale_review.consistency import verify_paths
from scale_review.errors import DivergenceError, LeaseError, ReplayError, SpecError
from scale_review.estimator import numpy_unclamped_replicas, unclamped_replicas
from scale_review.ledger import Ledger
from scale_review.leases import LeaseManager, replay_quota_events
from scale_review.propagation import VersionedPlan
from scale_review.replay import recover_and_replay, verify_ledger_replay
from scale_review.sample_data import build_sample_spec, write_sample_bundle
from scale_review.spec import load_spec_dict, load_spec_file
from scale_review.workbench import RoundRunner


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "spec.json"


class SpecValidationTests(unittest.TestCase):
    def test_valid_sample_loads(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        self.assertEqual(spec.service_ids, ["alpha", "beta", "gamma"])
        self.assertEqual(spec.minute_count, 18)

    def test_invalid_file_reports_first_field(self) -> None:
        with self.assertRaises(SpecError) as ctx:
            load_spec_file(ROOT / "configs" / "invalid-spec.json")
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "lease_policy.renew_interval_ms")

    def test_negative_request_reports_array_field(self) -> None:
        data = build_sample_spec(minute_count=4)
        data["services"][0]["requests_per_minute"][1] = -1
        with self.assertRaises(SpecError) as ctx:
            load_spec_dict(data)
        self.assertEqual(
            ctx.exception.field, "services[0].requests_per_minute[1]"
        )

    def test_duplicate_service_id(self) -> None:
        data = build_sample_spec(minute_count=3)
        data["services"][1]["service_id"] = data["services"][0]["service_id"]
        with self.assertRaises(SpecError) as ctx:
            load_spec_dict(data)
        self.assertIn("duplicate", ctx.exception.message)
        self.assertTrue(ctx.exception.field.endswith("service_id"))


class AlgorithmParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec_file(SPEC_PATH)

    def test_numpy_fixed_point_matches_scalar(self) -> None:
        values = [0, 1, 9, 10, 21, 99]
        vector = numpy_unclamped_replicas(values, 10, 1000)
        scalar = [unclamped_replicas(value, 10) for value in values]
        self.assertEqual(vector, scalar)

    def test_paths_and_canonical_bytes_match(self) -> None:
        full = run_full_path(self.spec)
        incremental = run_incremental_path(self.spec)
        full_allocations, inc_allocations = verify_paths(
            self.spec, full, incremental
        )
        self.assertEqual(
            conclusion_bytes(self.spec, full, full_allocations),
            conclusion_bytes(self.spec, incremental, inc_allocations),
        )
        selection = select_path(self.spec)
        self.assertEqual(selection.selected_path, "incremental")

    def test_service_order_does_not_change_bytes(self) -> None:
        reordered = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        reordered["services"] = list(reversed(reordered["services"]))
        spec_reordered = load_spec_dict(reordered)
        first = conclusion_bytes(self.spec, run_full_path(self.spec))
        second = conclusion_bytes(
            spec_reordered, run_full_path(spec_reordered)
        )
        self.assertEqual(first, second)

    def test_parallel_minute_evaluation_is_stable(self) -> None:
        jobs = [
            (service_id, minute)
            for service_id in self.spec.service_ids
            for minute in range(self.spec.minute_count)
        ]

        def evaluate(job: tuple[str, int]) -> bytes:
            service_id, minute = job
            service = self.spec.service(service_id)
            return canonical_bytes(
                compute_full_minute(self.spec, service, minute).comparable_dict()
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {job: pool.submit(evaluate, job) for job in reversed(jobs)}
            parallel = {job: future.result() for job, future in futures.items()}
        serial = {job: evaluate(job) for job in jobs}
        self.assertEqual(parallel, serial)

    def test_tie_boundary_uses_stable_winner(self) -> None:
        config = replace(
            self.spec.algorithm,
            path_switch_length=16,
            effort_tie_boundary=10,
            tie_break_path="full",
        )
        spec = replace(self.spec, algorithm=config)
        selection = select_path(spec, 16)
        self.assertEqual(selection.reason, "estimated_effort_within_explicit_tie_boundary")
        self.assertEqual(selection.selected_path, "full")


class DivergenceTests(unittest.TestCase):
    def test_divergence_locates_first_field_and_retains_views(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        full = run_full_path(spec)
        incremental = run_incremental_path(spec)
        alpha_minutes = list(incremental.minutes["alpha"])
        alpha_minutes[3] = replace(alpha_minutes[3], unclamped_replicas=999)
        bad_incremental = replace(
            incremental, minutes={**incremental.minutes, "alpha": tuple(alpha_minutes)}
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DivergenceError) as ctx:
                verify_paths(
                    spec, full, bad_incremental, artifact_dir=Path(tmp)
                )
            error = ctx.exception
            self.assertEqual(error.code, "E_DIVERGE")
            self.assertEqual(error.details["service_id"], "alpha")
            self.assertEqual(error.details["minute"], 3)
            self.assertEqual(error.details["field"], "unclamped_replicas")
            self.assertIn("input_views", error.details)
            self.assertTrue((Path(tmp) / "e_diverge_input_views.json").exists())


class LeaseTests(unittest.TestCase):
    def test_optimistic_version_rejection_does_not_extend(self) -> None:
        manager = LeaseManager(1, {"a": 1}, 500)
        lease = manager.grant("L1", "a", batch_id="B", task_id="T")
        assert lease is not None
        original_expiry = lease.expires_at_monotonic_ms
        with self.assertRaises(LeaseError) as ctx:
            manager.renew("L1", lease.version + 10)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        current = manager.get_lease("L1")
        self.assertEqual(current.version, 1)
        self.assertEqual(current.expires_at_monotonic_ms, original_expiry)

    def test_real_expiry_and_unique_conflict_verdict(self) -> None:
        manager = LeaseManager(2, {"a": 1, "b": 1}, 80)
        manager.grant("L1", "a", batch_id="B", task_id="T1")
        manager.grant("L2", "b", batch_id="B", task_id="T2")
        time.sleep(0.13)
        revoked = manager.revoke_batch("B")
        expired = manager.reap_expired()
        self.assertTrue(all(event["effective"] for event in revoked))
        self.assertEqual(expired, [])
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["global_remaining"], 2)
        self.assertEqual(snapshot["owner_set"], [])
        releases = [
            event
            for event in manager.events()
            if event["event_type"] == "quota_released" and event["effective"]
        ]
        self.assertEqual(len(releases), 2)
        self.assertTrue(all(event["source"] == "expiry" for event in releases))

    def test_event_replay_is_order_independent(self) -> None:
        manager = LeaseManager(1, {"a": 1}, 1000)
        manager.grant("L1", "a", batch_id="B", task_id="T")
        time.sleep(0.01)
        manager.revoke_batch("B")
        events = manager.events()
        first = replay_quota_events(1, {"a": 1}, events)
        second = replay_quota_events(
            1, {"a": 1}, list(reversed(events))
        )
        self.assertEqual(first, second)
        self.assertEqual(first["conflicts"], [])
        self.assertEqual(first["owner_set"], [])


class SchedulerAndPropagationTests(unittest.TestCase):
    def test_parallel_round_limits_and_wait_spans(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.sqlite3")
            runner = RoundRunner(spec, ledger)
            result = runner.run_round(7)
            self.assertLessEqual(result["max_observed_inflight"], 2)
            self.assertTrue(result["wait_spans"])
            self.assertTrue(all(span["waited_ms"] >= 0 for span in result["wait_spans"]))
            self.assertLessEqual(
                result["quota_snapshot"]["global_held"],
                spec.budget.global_replica_budget,
            )
            ledger.close()

    def test_append_and_budget_preserve_unaffected_bytes(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        plan = VersionedPlan(spec)
        old_demand = dict(plan.demand_bytes)
        old_allocation = dict(plan.allocation_bytes)
        batch = json.loads(
            (ROOT / "samples/request_batches/batch-001.json").read_text()
        )["requests"]
        propagation = plan.append_minutes(batch)
        self.assertEqual(plan.version, 2)
        self.assertEqual(propagation["affected_minutes"], [18, 19, 20])
        self.assertTrue(all(plan.demand_bytes[k] == v for k, v in old_demand.items()))
        self.assertTrue(
            all(plan.allocation_bytes[k] == v for k, v in old_allocation.items())
        )

        before_budget = dict(plan.allocation_bytes)
        budget_report = plan.update_budget(global_replica_budget=12)
        self.assertNotIn(0, budget_report["affected_minutes"])
        self.assertEqual(plan.allocation_bytes[(0, "alpha")], before_budget[(0, "alpha")])
        self.assertTrue(all(plan.demand_bytes[k] == v for k, v in old_demand.items()))

    def test_factor_update_preserves_nondirty_minutes(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        plan = VersionedPlan(spec)
        before = dict(plan.demand_bytes)
        report = plan.update_scaling_factor(1010)
        dirty = set(report["affected_minutes"])
        for key, value in before.items():
            if key[1] not in dirty:
                self.assertEqual(plan.demand_bytes[key], value)
        self.assertTrue(plan.audit_algorithm_paths()["equal"])


class LedgerRecoveryTests(unittest.TestCase):
    def test_snapshot_digest_mismatch_replaces_whole_file(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = Ledger(root / "ledger.sqlite3")
            runner = RoundRunner(spec, ledger)
            runner.run_round(0)
            snapshot_path = root / "snapshot.json"
            ledger.write_snapshot(snapshot_path)
            envelope = json.loads(snapshot_path.read_text())
            envelope["content_digest"] = "0" * 64
            snapshot_path.write_text(json.dumps(envelope))
            replaced, did_replace, reason = ledger.open_or_replace_snapshot(
                snapshot_path
            )
            self.assertTrue(did_replace)
            self.assertEqual(reason, "snapshot_digest_mismatch")
            self.assertTrue(replaced["content_digest"] != "0" * 64)
            ledger.close()

    def test_replay_detects_field_mismatch(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ledger.sqlite3"
            ledger = Ledger(db_path)
            runner = RoundRunner(spec, ledger)
            runner.run_round(1)
            ledger.close()

            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT id, canonical_json FROM minute_results "
                "WHERE service_id='alpha' AND minute=1 AND record_type='allocation'"
            ).fetchone()
            payload = json.loads(row[1])
            payload["allocated_replicas"] = int(payload["allocated_replicas"]) + 1
            conn.execute(
                "UPDATE minute_results SET canonical_json=? WHERE id=?",
                (json.dumps(payload, sort_keys=True), row[0]),
            )
            conn.commit()
            conn.close()

            ledger = Ledger(db_path)
            with self.assertRaises(ReplayError) as ctx:
                verify_ledger_replay(ledger, spec)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertIn("allocated_replicas", ctx.exception.field)
            self.assertEqual(ctx.exception.details["round_seq"], 1)
            ledger.close()

    def test_unknown_attempt_is_released_and_replay_succeeds(self) -> None:
        spec = load_spec_file(SPEC_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.sqlite3")
            runner = RoundRunner(spec, ledger)
            runner.run_round(2)
            ledger.begin_attempt(
                "attempt-1",
                "unfinished-round",
                3,
                1,
                {"lease_id": "tentative", "global_slots": [0], "version": 1},
            )
            result = recover_and_replay(ledger, spec)
            self.assertEqual(len(result["recovered_attempts"]), 1)
            self.assertEqual(result["recovered_attempts"][0]["status"], "unknown")
            self.assertTrue(result["replay"]["rounds"])
            self.assertEqual(result["next_minute"], 3)
            ledger.close()


if __name__ == "__main__":
    unittest.main()
