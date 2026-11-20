from __future__ import annotations

import copy
import json
import stat
import tempfile
import unittest
from pathlib import Path

from meshgov.budget import BudgetSnapshot
from meshgov.circuit import CircuitRuntime
from meshgov.errors import SnapshotError
from meshgov.ledger import Ledger
from meshgov.rules import parse_rules
from meshgov.selector import SelectionPlanner
from meshgov.snapshot import SnapshotStore, diff_snapshots, load_snapshot
from meshgov.viewmodel import build_view_model
from tests.helpers import instance, make_requests, make_rules


def two_versions():
    def document(timeout: int, retries: int):
        return {
            "schema_version": 1,
            "limits": {"global_concurrency": 2},
            "services": [
                {
                    "name": "alpha",
                    "concurrency_quota": 1,
                    "instances": [instance("a", 28001, 1)],
                }
            ],
            "routes": [
                {
                    "prefix": "/api/alpha",
                    "service": "alpha",
                    "timeout_ms": timeout,
                    "max_retries": retries,
                }
            ],
        }

    old = parse_rules(json.dumps(document(100, 1)).encode(), "<old>")
    new = parse_rules(json.dumps(document(200, 2)).encode(), "<new>")
    return old, new


class SnapshotTests(unittest.TestCase):
    def test_publish_replaces_latest_wholesale_and_keeps_old_file_readonly(self) -> None:
        old_rules, new_rules = two_versions()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            ledger = Ledger(directory / "mesh.db")
            try:
                store = SnapshotStore(directory / "snapshots")
                old = {
                    "schema_version": 1,
                    "snapshot_id": "snap-old",
                    "round_id": "round-old",
                    "rule_version": old_rules.rules_version,
                    "complete": True,
                    "rules": old_rules.raw_document,
                    "summary": {"requests_total": 2},
                    "requests": [],
                }
                new = {
                    "schema_version": 1,
                    "snapshot_id": "snap-new",
                    "round_id": "round-new",
                    "rule_version": new_rules.rules_version,
                    "complete": True,
                    "rules": new_rules.raw_document,
                    "summary": {"requests_total": 3},
                    "requests": [],
                }
                old_path = store.publish(old, ledger)
                old_bytes = old_path.read_bytes()
                old_mode = old_path.stat().st_mode
                new_path = store.publish(new, ledger)
                self.assertEqual(load_snapshot(store.latest_path)["snapshot_id"], "snap-new")
                self.assertEqual(old_path.read_bytes(), old_bytes)
                self.assertEqual(old_path.stat().st_mode, old_mode)
                self.assertEqual(old_mode & 0o222, 0)
                self.assertNotEqual(old_path, new_path)
                self.assertEqual(store.publish(new, ledger), new_path)
                diff = diff_snapshots(old, new)
                self.assertTrue(diff["version_changed"])
                self.assertEqual(diff["route_diff"]["changed_prefixes"], ["/api/alpha"])
                self.assertTrue(diff["snapshot_replaced_wholesale"])
            finally:
                ledger.close()

    def test_same_round_different_snapshot_is_rejected(self) -> None:
        rules = make_rules()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            ledger = Ledger(directory / "mesh.db")
            try:
                store = SnapshotStore(directory / "snapshots")
                first = {
                    "schema_version": 1,
                    "snapshot_id": "snap-one",
                    "round_id": "round-one",
                    "rule_version": rules.rules_version,
                    "complete": True,
                    "rules": rules.raw_document,
                }
                store.publish(first, ledger)
                second = dict(first, snapshot_id="snap-two")
                with self.assertRaises(SnapshotError) as raised:
                    store.publish(second, ledger)
                self.assertEqual(raised.exception.code, "E_SNAPSHOT")
            finally:
                ledger.close()

    def test_injected_snapshot_failure_has_business_code(self) -> None:
        rules = make_rules()
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(Path(temp) / "mesh.db")
            try:
                store = SnapshotStore(Path(temp) / "snapshots", injected_failure=True)
                with self.assertRaises(SnapshotError) as raised:
                    store.publish(
                        {
                            "snapshot_id": "snap-x",
                            "round_id": "round-x",
                            "rule_version": rules.rules_version,
                            "rules": rules.raw_document,
                        },
                        ledger,
                    )
                self.assertEqual(raised.exception.code, "E_SNAPSHOT")
                self.assertEqual(raised.exception.field_path, "$.snapshot")
            finally:
                ledger.close()


class RecoveryTests(unittest.TestCase):
    def test_recovery_reports_completed_and_unknown_items_without_exit_report(self) -> None:
        rules = make_rules()
        requests = make_requests(rules, 2)
        planner = SelectionPlanner(rules, requests)
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(Path(temp) / "mesh.db")
            try:
                ledger.begin_round("crash-round", rules, "input-sha")
                for request in requests:
                    ledger.queue_request("crash-round", request, 10.0)
                first = requests[0]
                second = requests[1]
                ledger.request_started("crash-round", first.request_id)
                selection = planner.get(first.request_id, 0)
                ledger.attempt_started(
                    "crash-round",
                    first,
                    1,
                    selection,
                    selected=True,
                    queue_ms=2.0,
                    global_active_at_start=1,
                    service_active_at_start=1,
                    permit_start_mono_ms=12.0,
                    circuit_admitted_mono_ms=12.0,
                    start_mono_ms=12.1,
                )
                ledger.attempt_finished(
                    "crash-round",
                    first.request_id,
                    1,
                    end_mono_ms=13.0,
                    result_status="success",
                    code=None,
                    detail="ok",
                    http_status=200,
                    selected=True,
                )
                ledger.attempt_permit_released(
                    "crash-round", first.request_id, 1, 13.1
                )
                ledger.request_finished(
                    "crash-round",
                    first.request_id,
                    final_status="success",
                    final_code=None,
                    attempts_used=1,
                    finished_mono_ms=13.2,
                )

                ledger.request_started("crash-round", second.request_id)
                second_selection = planner.get(second.request_id, 0)
                ledger.attempt_started(
                    "crash-round",
                    second,
                    1,
                    second_selection,
                    selected=True,
                    queue_ms=1.0,
                    global_active_at_start=1,
                    service_active_at_start=1,
                    permit_start_mono_ms=14.0,
                    circuit_admitted_mono_ms=14.0,
                    start_mono_ms=14.1,
                )
                ledger.persist_budget(
                    "crash-round",
                    BudgetSnapshot(
                        global_limit=2,
                        global_active=1,
                        global_peak=1,
                        services={"alpha": {"limit": 1, "active": 1, "peak": 1}},
                    ),
                )
                runtime = CircuitRuntime(
                    instance="alpha-a",
                    service="alpha",
                    host="127.0.0.1",
                    port=28001,
                    state="open",
                    consecutive_failures=2,
                    failure_threshold=2,
                    cooldown_ms=100,
                    opened_at_mono_ms=12.0,
                    cooldown_until_mono_ms=112.0,
                )
                ledger.save_circuit_state(runtime)
                report = ledger.recovery_report()
                self.assertFalse(report["exit_report_supported"])
                self.assertEqual(len(report["recovered_items"]), 1)
                self.assertEqual(len(report["unknown_items"]), 1)
                self.assertEqual(report["unknown_items"][0]["request_id"], "req-0002")
                self.assertEqual(len(report["unknown_attempts"]), 1)
                self.assertEqual(len(report["persisted_circuit_states"]), 1)
                self.assertEqual(report["persisted_quota_states"][0]["global"]["active"], 1)
            finally:
                ledger.close()


class ViewModelTests(unittest.TestCase):
    def test_view_model_uses_stable_order_and_single_version(self) -> None:
        rules = make_rules()
        snapshot = {
            "schema_version": 1,
            "snapshot_id": "snap-view",
            "round_id": "round-view",
            "rule_version": rules.rules_version,
            "created_at_utc": "2026-01-01T00:00:00Z",
            "complete": True,
            "rules": rules.raw_document,
            "limits": {
                "global_concurrency": 2,
                "services": {"alpha": 1},
            },
            "summary": {
                "status": "success",
                "requests_total": 1,
                "requests_succeeded": 1,
                "requests_failed": 0,
                "conclusion": "PASS",
            },
            "requests": [
                {
                    "request_id": "req-1",
                    "request_index": 0,
                    "status": "completed",
                    "rule_version": rules.rules_version,
                }
            ],
            "services": [
                {
                    "service": "alpha",
                    "concurrency_quota": 1,
                    "rule_version": rules.rules_version,
                    "instances": [
                        {
                            "instance": "a",
                            "port": 28001,
                            "weight": 1,
                            "targeted_count": 1,
                            "selected_count": 1,
                            "success_count": 1,
                            "failure_count": 0,
                            "fast_fail_count": 0,
                            "circuit_state": "closed",
                            "failure_threshold": 2,
                            "cooldown_ms": 100,
                            "rule_version": rules.rules_version,
                        }
                    ],
                }
            ],
            "budget": {
                "global": {"limit": 2, "active": 0, "peak": 1},
                "services": [{"service": "alpha", "limit": 1, "active": 0, "peak": 1}],
            },
            "circuit_events": [],
        }
        view = build_view_model(snapshot)
        self.assertEqual(view["progress"], {"completed": 1, "total": 1, "percent": 100.0})
        self.assertEqual(view["instances"][0]["service"], "alpha")
        self.assertEqual(view["routes"][0]["rule_version"], rules.rules_version)
        self.assertEqual(view["budget"]["global"]["utilization_percent"], 50.0)
        self.assertEqual(view["budget"]["services"][0]["utilization_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()

