from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from meshgov.ledger import Ledger
from meshgov.logging import JsonlLogger
from meshgov.rules import parse_rules
from meshgov.scheduler import Scheduler
from meshgov.snapshot import build_snapshot
from meshgov.transport import AttemptResponse
from meshgov.verify import verify_snapshot
from tests.helpers import instance, make_requests


class TrackingTransport:
    def __init__(
        self,
        *,
        fail_ports: set[int] | None = None,
        delays: dict[int, float] | None = None,
        fail_attempts: dict[int, int] | None = None,
    ) -> None:
        self.fail_ports = fail_ports or set()
        self.delays = delays or {}
        self.fail_attempts = fail_attempts or {}
        self.calls: dict[tuple[str, int], int] = {}
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def request(self, target: Any, request: Any, attempt_no: int) -> AttemptResponse:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls[(request.request_id, attempt_no)] = (
                self.calls.get((request.request_id, attempt_no), 0) + 1
            )
        try:
            time.sleep(self.delays.get(target.port, 0.001))
            fail = target.port in self.fail_ports
            fail_limit = self.fail_attempts.get(target.port)
            if fail_limit is not None and attempt_no <= fail_limit:
                fail = True
            if fail:
                return AttemptResponse(False, "E_UPSTREAM", "fake failure", 503)
            return AttemptResponse(True, None, "ok", 200)
        finally:
            with self.lock:
                self.active -= 1


def rules_with(
    *,
    instances: list[dict[str, Any]],
    quota: int = 2,
    global_limit: int = 3,
    retries: int = 2,
):
    document = {
        "schema_version": 1,
        "limits": {"global_concurrency": global_limit},
        "services": [
            {
                "name": "alpha",
                "concurrency_quota": quota,
                "instances": instances,
            }
        ],
        "routes": [
            {
                "prefix": "/api/alpha",
                "service": "alpha",
                "timeout_ms": 100,
                "max_retries": retries,
            }
        ],
    }
    return parse_rules(json.dumps(document).encode(), "<test>")


def run_round(
    directory: Path,
    rules: Any,
    transport: Any,
    *,
    count: int,
    round_id: str,
    ledger: Ledger | None = None,
) -> tuple[dict[str, Any], Any, Ledger]:
    requests = make_requests(rules, count)
    owns_ledger = ledger is None
    ledger = ledger or Ledger(directory / "mesh.db")
    logger = JsonlLogger(directory / f"{round_id}.jsonl")
    scheduler = Scheduler(rules, requests, transport, ledger, logger)
    scheduler.run(round_id, input_sha256="test-input")
    snapshot = build_snapshot(
        round_id,
        ledger,
        rules,
        rules_changed_during_round=False,
        observed_rule_version=rules.rules_version,
    )
    logger.close()
    if owns_ledger:
        # Keep the ledger open for snapshot queries and close it on failure.
        pass
    return snapshot, transport, ledger


class SchedulerVerificationTests(unittest.TestCase):
    def test_retry_then_success_and_budget_evidence(self) -> None:
        rules = rules_with(
            instances=[instance("solo", 28001, 1, threshold=10, cooldown_ms=100)],
            quota=1,
            global_limit=1,
            retries=2,
        )
        transport = TrackingTransport(fail_attempts={28001: 2})
        with tempfile.TemporaryDirectory() as temp:
            snapshot, _, ledger = run_round(
                Path(temp), rules, transport, count=2, round_id="retry"
            )
            try:
                verification = verify_snapshot(snapshot)
                self.assertEqual(verification["status"], "pass", verification)
                self.assertEqual(
                    [attempt["attempt_no"] for attempt in snapshot["attempts"]],
                    [1, 2, 3, 1, 2, 3],
                )
                self.assertTrue(
                    all(request["final_status"] == "success" for request in snapshot["requests"])
                )
                self.assertEqual(snapshot["summary"]["retries_used"], 4)
                self.assertTrue(all("queue_ms" in attempt for attempt in snapshot["attempts"]))
                self.assertTrue(
                    all(attempt["start_mono_ms"] <= attempt["end_mono_ms"] for attempt in snapshot["attempts"])
                )
            finally:
                ledger.close()

    def test_parallel_round_never_exceeds_budgets_and_circuit_fast_fails(self) -> None:
        rules = rules_with(
            instances=[
                instance("good", 28001, 1, threshold=2, cooldown_ms=50),
                instance("bad", 28002, 1, threshold=1, cooldown_ms=500),
            ],
            quota=2,
            global_limit=2,
            retries=1,
        )
        transport = TrackingTransport(
            fail_ports={28002},
            delays={28001: 0.02, 28002: 0.001},
        )
        with tempfile.TemporaryDirectory() as temp:
            snapshot, _, ledger = run_round(
                Path(temp), rules, transport, count=20, round_id="circuit"
            )
            try:
                verification = verify_snapshot(snapshot)
                self.assertEqual(verification["status"], "pass", verification)
                self.assertLessEqual(transport.peak, 2)
                self.assertLessEqual(snapshot["budget"]["global"]["peak"], 2)
                self.assertLessEqual(
                    snapshot["summary"]["service_concurrency"][0]["peak"], 2
                )
                self.assertGreater(snapshot["summary"]["circuit_fast_failures"], 0)
                self.assertTrue(
                    any(event["event"] == "opened" and event["port"] == 28002 for event in snapshot["circuit_events"])
                )
                fast_fails = [row for row in snapshot["attempts"] if row["code"] == "E_CIRCUIT"]
                self.assertTrue(fast_fails)
                self.assertTrue(all(not row["selected"] for row in fast_fails))
            finally:
                ledger.close()

    def test_target_sequence_is_repeatable_under_parallelism(self) -> None:
        rules = rules_with(
            instances=[
                instance("a", 28001, 3, threshold=10, cooldown_ms=100),
                instance("b", 28002, 2, threshold=10, cooldown_ms=100),
            ],
            quota=2,
            global_limit=3,
            retries=1,
        )
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            ledger = Ledger(directory / "mesh.db")
            try:
                snapshot_one, _, _ = run_round(
                    directory,
                    rules,
                    TrackingTransport(delays={28001: 0.006, 28002: 0.004}),
                    count=18,
                    round_id="det-one",
                    ledger=ledger,
                )
                snapshot_two, _, _ = run_round(
                    directory,
                    rules,
                    TrackingTransport(delays={28002: 0.006, 28001: 0.004}),
                    count=18,
                    round_id="det-two",
                    ledger=ledger,
                )
                sequence_one = [
                    (row["request_id"], row["attempt_no"], row["target_instance"])
                    for row in snapshot_one["attempts"]
                ]
                sequence_two = [
                    (row["request_id"], row["attempt_no"], row["target_instance"])
                    for row in snapshot_two["attempts"]
                ]
                self.assertEqual(sequence_one, sequence_two)
            finally:
                ledger.close()

    def test_verifier_reports_stable_first_deviation(self) -> None:
        rules = rules_with(
            instances=[instance("solo", 28001, 1, threshold=10, cooldown_ms=100)],
            quota=1,
            global_limit=1,
            retries=2,
        )
        transport = TrackingTransport(fail_attempts={28001: 1})
        with tempfile.TemporaryDirectory() as temp:
            snapshot, _, ledger = run_round(
                Path(temp), rules, transport, count=3, round_id="tamper"
            )
            try:
                self.assertEqual(verify_snapshot(snapshot)["status"], "pass")

                final_tampered = copy.deepcopy(snapshot)
                last_first_request = [
                    row for row in final_tampered["attempts"] if row["request_id"] == "req-0001"
                ][-1]
                last_first_request["result_status"] = "failure"
                result = verify_snapshot(final_tampered)
                self.assertEqual(result["code"], "E_VERIFY")
                self.assertEqual(result["first_deviation"]["request_id"], "req-0001")
                self.assertEqual(result["first_deviation"]["check"], "final.success_trace")

                concurrency_tampered = copy.deepcopy(snapshot)
                concurrency_tampered["limits"]["global_concurrency"] = 0
                result = verify_snapshot(concurrency_tampered)
                self.assertIn("concurrency.global", {item["check"] for item in result["violations"]})

                circuit_tampered = copy.deepcopy(snapshot)
                first_attempt = circuit_tampered["attempts"][0]
                circuit_tampered["circuit_events"].append(
                    {
                        "service": first_attempt["service"],
                        "port": first_attempt["target_port"],
                        "event": "opened",
                        "opened_at_mono_ms": first_attempt["start_mono_ms"] - 1,
                        "cooldown_until_mono_ms": first_attempt["start_mono_ms"] + 10000,
                    }
                )
                result = verify_snapshot(circuit_tampered)
                self.assertIn(
                    "circuit.cooldown_dispatch",
                    {item["check"] for item in result["violations"]},
                )

                retry_tampered = copy.deepcopy(snapshot)
                retry_tampered["requests"][0]["max_retries"] = 0
                result = verify_snapshot(retry_tampered)
                self.assertIn("retry.budget", {item["check"] for item in result["violations"]})

                selection_tampered = copy.deepcopy(snapshot)
                selection_tampered["services"][0]["instances"][0]["weight"] = 0
                result = verify_snapshot(selection_tampered)
                self.assertIn(
                    "selection.zero_weight",
                    {item["check"] for item in result["violations"]},
                )
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
