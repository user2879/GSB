from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from meshwork.engine import MeshRuntime
from meshwork.ledger import Ledger
from meshwork.logging_utils import StructuredLogger
from meshwork.model import parse_requests, parse_spec
from meshwork.replay import compare_ledgers
from meshwork.snapshot import SnapshotStore
from meshwork.viewmodel import build_view_model
from tests.helpers import behavior, free_ports, instance, make_spec


class IntegrationTests(unittest.TestCase):
    def test_parallel_budget_limits_and_view_refresh_are_content_stable(self) -> None:
        ports = free_ports(3)
        raw = make_spec(
            [
                {
                    "id": "alpha",
                    "quota": 1,
                    "instances": [
                        instance(
                            "alpha-a",
                            port=ports[0],
                            weight=2,
                            behavior_value=behavior(
                                fast_ms=5,
                                slow_ms=80,
                                rules=[
                                    {
                                        "request_prefix": "slow",
                                        "outcome": "success",
                                        "latency_class": "slow",
                                    }
                                ],
                            ),
                        )
                    ],
                },
                {
                    "id": "beta",
                    "quota": 1,
                    "instances": [
                        instance(
                            "beta-a",
                            port=ports[1],
                            weight=1,
                            behavior_value=behavior(
                                fast_ms=5,
                                slow_ms=80,
                                rules=[
                                    {
                                        "request_prefix": "slow",
                                        "outcome": "success",
                                        "latency_class": "slow",
                                    }
                                ],
                            ),
                        )
                    ],
                },
            ],
            global_concurrency=2,
            timeout_ms=300,
            backoffs=[2],
        )
        spec = parse_spec(raw)
        requests = parse_requests(
            [
                {"id": f"slow-a-{index}", "service": "alpha", "payload": {"n": index}}
                for index in range(6)
            ]
            + [
                {"id": f"slow-b-{index}", "service": "beta", "payload": {"n": index}}
                for index in range(6)
            ],
            spec.services.keys(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = Ledger(root / "ledger.db", reset=True)
            report: dict = {}
            runtime = MeshRuntime(
                spec,
                spec.active_revision(),
                ledger,
                SnapshotStore(root / "snapshots"),
                StructuredLogger(root / "log.jsonl"),
                "integration-budget",
            )
            try:
                runtime.start()
                report = runtime.run_requests(requests)
            finally:
                runtime.stop()
            budget = report["budget"]
            self.assertLessEqual(budget["peak_global"], 2)
            self.assertLessEqual(budget["peak_service"]["alpha"], 1)
            self.assertLessEqual(budget["peak_service"]["beta"], 1)
            self.assertEqual(budget["global_inflight"], 0)
            attempts = ledger.latest_attempts()
            self.assertTrue(attempts)
            for row in attempts:
                self.assertLessEqual(row["global_inflight"], 2)
                self.assertLessEqual(row["service_inflight"], 1)
                self.assertGreaterEqual(row["wait_end_mono_ms"], row["wait_start_mono_ms"])
            before = ledger.content_digest()
            view1 = build_view_model(
                spec,
                spec.active_revision(),
                ledger,
                SnapshotStore(root / "snapshots"),
                runtime.runtime_state(),
            )
            view2 = build_view_model(
                spec,
                spec.active_revision(),
                ledger,
                SnapshotStore(root / "snapshots"),
                runtime.runtime_state(),
            )
            self.assertEqual(before, ledger.content_digest())
            self.assertTrue(view1["ledger_refresh"]["unchanged"])
            self.assertEqual(view1["services"], view2["services"])
            self.assertEqual(view1["stats"]["rankings"], view2["stats"]["rankings"])
            ledger.close()

    def test_retry_backoff_does_not_hold_global_budget(self) -> None:
        ports = free_ports(2)
        error_behavior = behavior(success=0, slow=0, error=100)
        raw = make_spec(
            [
                {
                    "id": "retry",
                    "quota": 1,
                    "instances": [
                        instance("retry-a", port=ports[0], behavior_value=error_behavior)
                    ],
                },
                {
                    "id": "other",
                    "quota": 1,
                    "instances": [
                        instance("other-a", port=ports[1], behavior_value=behavior())
                    ],
                },
            ],
            global_concurrency=1,
            timeout_ms=300,
            backoffs=[220],
            failure_threshold=20,
        )
        spec = parse_spec(raw)
        requests = parse_requests(
            [
                {"id": "retry-request", "service": "retry", "payload": {}},
                {"id": "other-request", "service": "other", "payload": {}},
            ],
            spec.services.keys(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = Ledger(root / "ledger.db", reset=True)
            runtime = MeshRuntime(
                spec,
                spec.active_revision(),
                ledger,
                SnapshotStore(root / "snapshots"),
                StructuredLogger(root / "log.jsonl"),
                "integration-backoff",
            )
            try:
                runtime.start()
                report = runtime.run_requests(requests)
            finally:
                runtime.stop()
            results = {row["request_id"]: row for row in report["results"]}
            self.assertEqual(results["retry-request"]["final_outcome"], "exhausted")
            self.assertEqual(results["other-request"]["final_outcome"], "success")
            attempts = ledger.latest_attempts()
            retry_attempts = [row for row in attempts if row["service"] == "retry"]
            other_attempts = [row for row in attempts if row["service"] == "other"]
            self.assertEqual(len(retry_attempts), 2)
            self.assertTrue(other_attempts)
            gap_start = int(retry_attempts[0]["finished_mono_ms"])
            gap_end = int(retry_attempts[1]["started_mono_ms"])
            other_start = int(other_attempts[0]["started_mono_ms"])
            self.assertGreaterEqual(other_start, gap_start)
            self.assertLessEqual(other_start, gap_end)
            self.assertGreaterEqual(gap_end - gap_start, 180)
            self.assertTrue(all(row["budget_used"] for row in retry_attempts))
            ledger.close()

    def test_circuit_trigger_fast_fail_and_half_open_recovery(self) -> None:
        ports = free_ports(1)
        raw = make_spec(
            [
                {
                    "id": "fragile",
                    "quota": 1,
                    "instances": [
                        instance(
                            "fragile-a",
                            port=ports[0],
                            behavior_value=behavior(
                                rules=[
                                    {
                                        "request_prefix": "cb-fail",
                                        "outcome": "error",
                                        "latency_class": "fast",
                                    },
                                    {
                                        "request_prefix": "cb-ok",
                                        "outcome": "success",
                                        "latency_class": "fast",
                                    },
                                ]
                            ),
                        )
                    ],
                }
            ],
            global_concurrency=1,
            timeout_ms=300,
            backoffs=[4, 8],
            failure_threshold=2,
            cooldown_ms=100,
            half_open_probes=1,
        )
        spec = parse_spec(raw)
        fail_requests = parse_requests(
            [{"id": "cb-fail-1", "service": "fragile", "payload": {}}], spec.services.keys()
        )
        ok_requests = parse_requests(
            [{"id": "cb-ok-1", "service": "fragile", "payload": {}}], spec.services.keys()
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = Ledger(root / "ledger.db", reset=True)
            runtime = MeshRuntime(
                spec,
                spec.active_revision(),
                ledger,
                SnapshotStore(root / "snapshots"),
                StructuredLogger(root / "log.jsonl"),
                "integration-circuit",
            )
            try:
                runtime.start()
                fail_report = runtime.run_requests(fail_requests)
                self.assertEqual(fail_report["results"][0]["error_code"], "E_EXHAUST")
                breaker = runtime.breakers.get("fragile-a")
                self.assertEqual(breaker.state.value, "OPEN")
                attempts = ledger.latest_attempts()
                self.assertEqual(len(attempts), 3)
                self.assertEqual(attempts[-1]["outcome"], "circuit_open")
                self.assertFalse(attempts[-1]["budget_used"])
                time.sleep(0.13)
                ok_report = runtime.run_requests(ok_requests)
                self.assertEqual(ok_report["results"][0]["final_outcome"], "success")
                self.assertEqual(breaker.state.value, "CLOSED")
                states = [(item.previous, item.current) for item in breaker.transitions]
                self.assertIn(("CLOSED", "OPEN"), states)
                self.assertIn(("OPEN", "HALF_OPEN"), states)
                self.assertIn(("HALF_OPEN", "CLOSED"), states)
            finally:
                runtime.stop()
            ledger.close()

    def test_full_replay_is_consistent_on_a_fresh_ledger(self) -> None:
        ports = free_ports(1)
        raw = make_spec(
            [
                {
                    "id": "svc",
                    "quota": 1,
                    "instances": [instance("svc-a", port=ports[0], behavior_value=behavior())],
                }
            ],
            global_concurrency=2,
            timeout_ms=300,
            backoffs=[2],
        )
        spec = parse_spec(raw)
        requests = parse_requests(
            [
                {"id": f"replay-{index}", "service": "svc", "payload": {"n": index}}
                for index in range(5)
            ],
            spec.services.keys(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = Ledger(root / "first.db", reset=True)
            runtime = MeshRuntime(
                spec,
                spec.active_revision(),
                first,
                SnapshotStore(root / "snap1"),
                StructuredLogger(root / "log1.jsonl"),
                "replay-first",
            )
            try:
                runtime.start()
                runtime.run_requests(requests)
            finally:
                runtime.stop()
            second = Ledger(root / "second.db", reset=True)
            runtime2 = MeshRuntime(
                spec,
                spec.active_revision(),
                second,
                SnapshotStore(root / "snap2"),
                StructuredLogger(root / "log2.jsonl"),
                "replay-second",
            )
            try:
                runtime2.start()
                runtime2.run_requests(requests)
                comparison = compare_ledgers(first, second)
            finally:
                runtime2.stop()
            self.assertEqual(comparison["status"], "consistent")
            self.assertEqual(comparison["compared_requests"], 5)
            first.close()
            second.close()

    def test_valid_snapshot_restores_budget_and_breaker_state_on_reopen(self) -> None:
        ports = free_ports(1)
        raw = make_spec(
            [
                {
                    "id": "svc",
                    "quota": 1,
                    "instances": [
                        instance(
                            "svc-a",
                            port=ports[0],
                            behavior_value=behavior(success=0, slow=0, error=100),
                        )
                    ],
                }
            ],
            global_concurrency=1,
            timeout_ms=300,
            backoffs=[2, 2],
            failure_threshold=1,
            cooldown_ms=5000,
            half_open_probes=1,
        )
        spec = parse_spec(raw)
        requests = parse_requests(
            [{"id": "snapshot-state", "service": "svc", "payload": {}}],
            spec.services.keys(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger_path = root / "ledger.db"
            snapshot_dir = root / "snapshots"
            ledger = Ledger(ledger_path, reset=True)
            runtime = MeshRuntime(
                spec,
                spec.active_revision(),
                ledger,
                SnapshotStore(snapshot_dir),
                StructuredLogger(root / "log1.jsonl"),
                "snapshot-state",
            )
            try:
                runtime.start()
                runtime.run_requests(requests)
                self.assertEqual(runtime.breakers.get("svc-a").state.value, "OPEN")
                runtime.write_snapshot("state")
            finally:
                runtime.stop()
            ledger.close()
            reopened = Ledger(ledger_path)
            reopened_runtime = MeshRuntime(
                spec,
                spec.active_revision(),
                reopened,
                SnapshotStore(snapshot_dir),
                StructuredLogger(root / "log2.jsonl"),
                "snapshot-state-reopen",
            )
            try:
                validation = reopened_runtime.validate_or_replace_snapshot()
                self.assertEqual(validation["status"], "valid")
                self.assertEqual(reopened_runtime.scheduler.snapshot()["global_grants"], 1)
                breaker = reopened_runtime.breakers.get("svc-a")
                self.assertEqual(breaker.state.value, "OPEN")
                self.assertGreater(breaker.cooldown_remaining_ms(), 0)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
