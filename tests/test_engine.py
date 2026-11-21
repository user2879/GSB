import unittest

from emergency_drill.engine import ALLOWED_TRANSITIONS, simulate
from emergency_drill.loader import load_bundle
from emergency_drill.sample_data import generate_samples
import os
import tempfile


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        generate_samples(self.tmp.name)
        self.plan, self.events = load_bundle(
            os.path.join(self.tmp.name, "plan_valid.json"),
            os.path.join(self.tmp.name, "events_valid.json"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_transition_table_is_explicit(self):
        self.assertNotIn("pending", ALLOWED_TRANSITIONS["succeeded"])
        self.assertEqual(ALLOWED_TRANSITIONS["running"], {"succeeded", "failed"})

    def test_failure_cascades_and_records_skip_trace(self):
        result = simulate(self.plan, self.events)
        isolate = result.outcomes["isolate"]
        self.assertEqual(isolate["state"], "failed")
        self.assertEqual(isolate["reason"], "switch_fail")
        for sid in ("failover", "verify", "recover"):
            self.assertEqual(result.outcomes[sid]["state"], "skipped")
            self.assertEqual(result.outcomes[sid]["upstream_failure"], "isolate")
        self.assertEqual(result.outcomes["recover"]["skip_trace"],
                         ["isolate", "failover", "verify", "recover"])
        self.assertEqual(result.stats["failed"], 1)
        self.assertEqual(result.stats["skipped"], 3)

    def test_logical_clock_monotonic_without_sleep(self):
        result = simulate(self.plan, self.events)
        clocks = [transition["clock_ms"] for transition in result.transitions]
        self.assertEqual(clocks, sorted(clocks))
        self.assertTrue(all(isinstance(value, int) for value in clocks))

    def test_switch_fail_fails_first_action_and_all_of_criterion(self):
        result = simulate(self.plan, self.events)
        self.assertEqual(result.outcomes["isolate"]["state"], "failed")
        self.assertEqual(result.outcomes["isolate"]["failed_actions"], ["a_block_traffic"])

    def test_deterministic_outputs_are_stable(self):
        first = simulate(self.plan, self.events)
        second = simulate(self.plan, self.events)
        self.assertEqual(first.outcomes, second.outcomes)
        self.assertEqual(first.transitions, second.transitions)


if __name__ == "__main__":
    unittest.main()
