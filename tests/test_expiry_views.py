from __future__ import annotations

import unittest

from config_diag.clock import ManualClock
from config_diag.engine import ConfigEngine
from config_diag.structured_log import NullLogger
from config_diag.subscriptions import SubscriptionSimulator
from config_diag.validation import load_bundle

from .support import SAMPLES, local_temp


def _run_expiry_scenario(tmp) -> dict:
    spec, clients, changes = load_bundle(SAMPLES)
    clock = ManualClock(1_000_000_000)
    engine = ConfigEngine(tmp, logger=NullLogger(), clock=clock)
    try:
        engine.initialize(spec, clients, now_ns=1_000_000_000)
        for operation in changes.operations:
            engine.apply_operation(operation, now_ns=1_000_000_000)
        clock.advance_ms(181)
        result = engine.collect_expired()
        result["statuses"] = {
            key: (record["status"], record["value"], record["version"])
            for key, record in engine.ledger.records().items()
        }
        result["active_keys"] = [item["key"] for item in engine.current_view()["active"]]
        result["second_round_reclaimed"] = engine.collect_expired()["reclaimed"]
        return result
    finally:
        engine.close()


class ExpiryAndViewTests(unittest.TestCase):
    def test_expiry_uses_monotonic_clock_and_invalidates_downstream_stably(self) -> None:
        with local_temp("expiry-a-") as tmp_a, local_temp("expiry-b-") as tmp_b:
            first = _run_expiry_scenario(tmp_a)
            second = _run_expiry_scenario(tmp_b)
        self.assertEqual(first["reclaimed"], ["cache.region_limit"])
        self.assertEqual(first["downstream"], ["cache.shard_span"])
        self.assertEqual(first["affected"], ["cache.region_limit", "cache.shard_span"])
        self.assertEqual(first["reclaimed"], second["reclaimed"])
        self.assertEqual(first["downstream"], second["downstream"])
        self.assertEqual(first["affected"], second["affected"])
        self.assertEqual(first["statuses"]["cache.region_limit"][0], "expired")
        self.assertIsNone(first["statuses"]["cache.region_limit"][1])
        self.assertEqual(first["statuses"]["cache.shard_span"][0], "invalid")
        self.assertIsNone(first["statuses"]["cache.shard_span"][1])
        self.assertNotIn("cache.region_limit", first["active_keys"])
        self.assertNotIn("cache.shard_span", first["active_keys"])
        self.assertEqual(first["second_round_reclaimed"], [])

    def test_stale_views_and_subscription_cache_are_whole_replacements(self) -> None:
        spec, clients, changes = load_bundle(SAMPLES)
        with local_temp("views-") as tmp:
            engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
            try:
                engine.initialize(spec, clients, now_ns=1_000_000_000)
                simulator = SubscriptionSimulator(engine, clients)
                first_poll = simulator.poll(1)
                old_view = engine.current_view()
                self.assertEqual(old_view["view_version"], 1)
                self.assertFalse(any(event["client_id"] == "" for event in simulator.stale_events))

                result = engine.apply_operation(changes.operations[0], now_ns=1_100_000_000)
                new_view = engine.current_view()
                self.assertEqual(new_view["view_version"], 2)
                views = engine.ledger.views()
                current = [item for item in views if item["status"] == "current"]
                stale = [item for item in views if item["status"] == "stale"]
                self.assertEqual(len(current), 1)
                self.assertEqual(current[0]["view_version"], 2)
                self.assertEqual([item["view_version"] for item in stale], [1])

                second_poll = simulator.poll(1)
                self.assertEqual(len(second_poll["stale_events"]), len(clients.clients))
                self.assertTrue(all(item["action"] == "whole_view_replace" for item in second_poll["stale_events"]))
                self.assertEqual(second_poll["statistics_source"], "current-view-only")
                current_records = {
                    item["key"]: item for item in new_view["active"]
                }
                for round_item in second_poll["rounds"]:
                    for client in round_item["clients"]:
                        for observation in client["observations"]:
                            self.assertEqual(
                                observation["content_digest"],
                                current_records[observation["key"]]["content_digest"],
                            )
                self.assertEqual(result["view_added"], [])
                self.assertEqual(result["view_deleted"], [])
                self.assertTrue(first_poll["rounds"])
            finally:
                engine.close()

    def test_polling_is_deterministic_for_fixed_integer_seeds(self) -> None:
        spec, clients, _ = load_bundle(SAMPLES)
        outputs = []
        for index in range(2):
            with local_temp(f"poll-{index}-") as tmp:
                engine = ConfigEngine(tmp, logger=NullLogger(), clock=ManualClock(1_000_000_000))
                try:
                    engine.initialize(spec, clients, now_ns=1_000_000_000)
                    outputs.append(SubscriptionSimulator(engine, clients).poll(3))
                finally:
                    engine.close()
        self.assertEqual(outputs[0]["rounds"], outputs[1]["rounds"])


if __name__ == "__main__":
    unittest.main()
