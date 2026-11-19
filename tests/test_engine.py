from __future__ import annotations

import json
import unittest

from srebudget.models import CircuitState, canonical_json

from .support import call, chain, drive, service, trace


class BudgetAndFastFailTests(unittest.TestCase):
    def setUp(self):
        self.chain = chain(
            [
                service(
                    "leaf",
                    ["root"],
                    ratio_bp=5000,
                    min_samples=2,
                    failure_threshold=10,
                ),
                service(
                    "root",
                    [],
                    ratio_bp=5000,
                    min_samples=2,
                    failure_threshold=10,
                ),
            ]
        )

    def test_budget_exhaustion_fast_fails_ancestors_without_sampling(self):
        stream = trace(
            [
                call(1000, ["leaf", "root"], "failure"),
                call(2000, ["leaf", "root"], "failure"),
            ],
            self.chain,
        )
        engine, _, decisions = drive(self.chain, stream)
        decision = decisions[1]
        self.assertEqual(decision.trigger_source, "leaf")
        self.assertEqual(decision.trigger_reason, "budget_exhausted")
        self.assertTrue(decision.fast_failed)

        by_service = {item.service: item for item in decision.services}
        self.assertEqual(by_service["leaf"].status, "consumed")
        self.assertEqual(by_service["leaf"].budget["window_samples"], 2)
        self.assertEqual(by_service["leaf"].budget["failures"], 2)
        self.assertEqual(by_service["leaf"].budget["budget_total"], 1)
        self.assertEqual(by_service["leaf"].budget["raw_remaining"], -1)
        self.assertEqual(by_service["leaf"].budget["remaining"], 0)
        self.assertTrue(by_service["leaf"].budget["exhausted"])

        self.assertEqual(by_service["root"].status, "fast_failed")
        self.assertFalse(by_service["root"].consumed)
        self.assertEqual(by_service["root"].budget["window_samples"], 1)
        self.assertEqual(by_service["root"].trigger_source, "leaf")
        self.assertEqual(by_service["root"].trigger_reason, "budget_exhausted")
        self.assertEqual(by_service["root"].propagation_depth, 1)

    def test_window_rollover_recovers_and_is_integer_only(self):
        one = chain(
            [
                service(
                    "svc",
                    [],
                    ratio_bp=5000,
                    min_samples=1,
                    window_hours=1,
                )
            ]
        )
        stream = trace(
            [call(0, ["svc"], "failure"), call(3_600_001, ["svc"], "success")],
            one,
        )
        engine, _, decisions = drive(one, stream)
        before = decisions[0].services[0].budget
        after = decisions[1].services[0].budget
        self.assertTrue(before["exhausted"])
        self.assertEqual(before["raw_remaining"], -1)
        self.assertFalse(after["exhausted"])
        self.assertEqual(after["window_samples"], 1)
        self.assertEqual(after["failures"], 0)
        self.assertEqual(after["budget_total"], 0)
        self.assertEqual(after["remaining"], 0)
        kinds = [event["kind"] for event in engine.events]
        self.assertIn("budget_exhausted", kinds)
        self.assertIn("budget_recovered", kinds)
        self.assertTrue(all(isinstance(value, int) for value in after.values() if not isinstance(value, bool)))

    def test_repeated_runs_are_byte_identical(self):
        stream = trace(
            [
                call(1000, ["leaf", "root"], "failure"),
                call(2000, ["leaf", "root"], "failure"),
                call(3000, ["leaf", "root"], "success"),
            ],
            self.chain,
        )
        _, _, first = drive(self.chain, stream)
        _, _, second = drive(self.chain, stream)
        left = canonical_json([item.as_dict() for item in first])
        right = canonical_json([item.as_dict() for item in second])
        self.assertEqual(left.encode("utf-8"), right.encode("utf-8"))


class CircuitStateTests(unittest.TestCase):
    def setUp(self):
        self.chain = chain(
            [
                service(
                    "svc",
                    ["caller"],
                    ratio_bp=10000,
                    min_samples=1,
                    failure_threshold=2,
                    cooldown_ms=1000,
                ),
                service(
                    "caller",
                    [],
                    ratio_bp=10000,
                    min_samples=1,
                    failure_threshold=10,
                ),
            ]
        )

    def test_open_half_open_close_and_probe_failure(self):
        stream = trace(
            [
                call(1000, ["svc", "caller"], "failure"),
                call(2000, ["svc", "caller"], "failure"),
                call(2500, ["svc", "caller"], "success"),
                call(3500, ["svc", "caller"], "success"),
                call(4000, ["svc", "caller"], "failure"),
                call(5000, ["svc", "caller"], "timeout"),
                call(6500, ["svc", "caller"], "success"),
            ],
            self.chain,
        )
        engine, _, decisions = drive(self.chain, stream)
        second = decisions[1].services[0]
        self.assertEqual(second.circuit["state"], "open")
        self.assertEqual(second.circuit["cooldown_remaining_ms"], 1000)
        self.assertEqual(second.circuit["open_count"], 1)

        cooling = decisions[2].services[0]
        self.assertEqual(cooling.status, "blocked")
        self.assertEqual(cooling.circuit["cooldown_remaining_ms"], 500)
        self.assertEqual(decisions[2].services[1].status, "fast_failed")

        half_open = decisions[3].services[0]
        self.assertEqual(half_open.status, "consumed")
        self.assertEqual(half_open.circuit["state"], "closed")
        self.assertEqual(half_open.circuit["consecutive_failures"], 0)
        self.assertEqual(decisions[3].services[1].status, "consumed")

        reopened = decisions[5].services[0]
        self.assertEqual(reopened.circuit["state"], "open")
        self.assertEqual(reopened.circuit["open_count"], 2)

        transitions = [
            event for event in engine.events if event["kind"] == "circuit_transition"
        ]
        self.assertEqual(
            [event["reason"] for event in transitions[:3]],
            [
                "failure_threshold_reached",
                "cooldown_elapsed",
                "half_open_probe_succeeded",
            ],
        )
        for event in transitions:
            self.assertIn("call_index", event)
            self.assertIn("consecutive_failures", event)
            self.assertIn("cooldown_remaining_ms", event)
            self.assertIn("reason", event)


if __name__ == "__main__":
    unittest.main()
