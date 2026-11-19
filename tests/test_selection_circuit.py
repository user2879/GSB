from __future__ import annotations

import unittest

from meshdiag.circuit import CircuitBreakerRegistry
from meshdiag.models import Instance, Service
from meshdiag.selection import select_instance


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.service = Service(
            "svc",
            2,
            (
                Instance("a", "127.0.0.1", 19001),
                Instance("b", "127.0.0.1", 19002),
                Instance("zero", "127.0.0.1", 19003),
            ),
        )

    def test_zero_weight_is_never_selected_and_distribution_is_explicit(self):
        counts = {"a": 0, "b": 0, "zero": 0}
        for index in range(6000):
            selected = select_instance(
                self.service, {"a": 2, "b": 1, "zero": 0}, f"req-{index}", 1
            )
            counts[selected.instance_id] += 1
        self.assertEqual(counts["zero"], 0)
        self.assertGreater(counts["a"], counts["b"])
        self.assertGreater(counts["b"], 0)
        self.assertAlmostEqual(counts["a"] / counts["b"], 2.0, delta=0.15)

    def test_selection_is_independent_of_input_mapping_order(self):
        first = select_instance(
            self.service, {"b": 1, "a": 2, "zero": 0}, "same-request", 2
        )
        second = select_instance(
            self.service, {"zero": 0, "a": 2, "b": 1}, "same-request", 2
        )
        self.assertEqual(first, second)


class FakeClock:
    def __init__(self):
        self.ns = 1_000_000_000

    def __call__(self):
        return self.ns

    def advance_ms(self, milliseconds):
        self.ns += milliseconds * 1_000_000


class CircuitBreakerTests(unittest.TestCase):
    def test_threshold_fast_fail_cooldown_and_successful_probe(self):
        clock = FakeClock()
        breakers = CircuitBreakerRegistry(clock)
        self.assertEqual(breakers.allow("a"), (True, "closed"))
        breakers.record_failure("a", threshold=2, cooldown_ms=1000)
        self.assertTrue(breakers.allow("a")[0])
        breakers.record_failure("a", threshold=2, cooldown_ms=1000)
        self.assertEqual(breakers.allow("a"), (False, "circuit_open"))
        clock.advance_ms(999)
        self.assertFalse(breakers.allow("a")[0])
        clock.advance_ms(1)
        self.assertEqual(breakers.allow("a"), (True, "circuit_probe"))
        self.assertEqual(breakers.allow("a"), (False, "circuit_probe_busy"))
        breakers.record_success("a")
        self.assertEqual(breakers.allow("a"), (True, "closed"))

    def test_failed_half_open_probe_reopens_cooldown(self):
        clock = FakeClock()
        breakers = CircuitBreakerRegistry(clock)
        breakers.record_failure("a", threshold=1, cooldown_ms=100)
        self.assertFalse(breakers.allow("a")[0])
        clock.advance_ms(100)
        self.assertTrue(breakers.allow("a")[0])
        breakers.record_failure("a", threshold=1, cooldown_ms=250)
        self.assertFalse(breakers.allow("a")[0])
        clock.advance_ms(249)
        self.assertFalse(breakers.allow("a")[0])
        clock.advance_ms(1)
        self.assertTrue(breakers.allow("a")[0])

    def test_success_resets_consecutive_failures(self):
        clock = FakeClock()
        breakers = CircuitBreakerRegistry(clock)
        breakers.record_failure("a", threshold=2, cooldown_ms=100)
        breakers.record_success("a")
        breakers.record_failure("a", threshold=2, cooldown_ms=100)
        self.assertTrue(breakers.allow("a")[0])


if __name__ == "__main__":
    unittest.main()
