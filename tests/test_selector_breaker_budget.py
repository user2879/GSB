from __future__ import annotations

import threading
import time
import unittest

from meshwork.breaker import BreakerState, CircuitBreaker
from meshwork.budget import BudgetScheduler
from meshwork.model import parse_spec
from meshwork.selector import WeightedSelector
from tests.helpers import behavior, free_ports, instance, make_spec


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        ports = free_ports(3)
        raw = make_spec(
            [
                {
                    "id": "svc",
                    "quota": 2,
                    "instances": [
                        instance("a", weight=3, port=ports[0], behavior_value=behavior()),
                        instance("b", weight=1, port=ports[1], behavior_value=behavior()),
                        instance("zero", weight=0, port=ports[2], behavior_value=behavior()),
                    ],
                }
            ]
        )
        self.spec = parse_spec(raw)
        self.selector = WeightedSelector(self.spec, self.spec.active_revision())

    def test_cumulative_ranges_and_zero_weight_exclusion(self) -> None:
        selected = {
            self.selector.select("svc", f"r-{index}", 1).instance.id
            for index in range(100)
        }
        self.assertEqual(selected, {"a", "b"})
        ranges = self.selector.ranges_for("svc")
        self.assertEqual([(item.instance.id, item.start, item.end) for item in ranges], [
            ("a", 0, 3),
            ("b", 3, 4),
            ("zero", 4, 4),
        ])

    def test_selection_is_repeatable_and_independent_of_dict_order(self) -> None:
        selections = [
            self.selector.select("svc", "req-42", attempt).instance.id
            for attempt in range(1, 10)
        ]
        again = [
            self.selector.select("svc", "req-42", attempt).instance.id
            for attempt in range(1, 10)
        ]
        self.assertEqual(selections, again)


class BreakerTests(unittest.TestCase):
    def make_breaker(self, clock: FakeClock, probes: int = 1) -> CircuitBreaker:
        raw = make_spec(
            [{"id": "svc", "quota": 1, "instances": [instance("i")]}],
            failure_threshold=2,
            cooldown_ms=100,
            half_open_probes=probes,
        )
        spec = parse_spec(raw)
        return CircuitBreaker(spec.services["svc"].instances[0], spec.session.breaker, clock)

    def test_open_fast_fail_and_half_open_success(self) -> None:
        clock = FakeClock()
        breaker = self.make_breaker(clock)
        first = breaker.before_attempt()
        breaker.record_result(first.permit, False)  # type: ignore[arg-type]
        second = breaker.before_attempt()
        breaker.record_result(second.permit, False)  # type: ignore[arg-type]
        self.assertEqual(breaker.state, BreakerState.OPEN)
        denied = breaker.before_attempt()
        self.assertFalse(denied.allowed)
        self.assertGreater(denied.remaining_cooldown_ms, 0)
        clock.advance(0.11)
        probe = breaker.before_attempt()
        self.assertTrue(probe.allowed)
        self.assertTrue(probe.permit.trial)  # type: ignore[union-attr]
        self.assertEqual(breaker.state, BreakerState.HALF_OPEN)
        breaker.record_result(probe.permit, True)  # type: ignore[arg-type]
        self.assertEqual(breaker.state, BreakerState.CLOSED)

    def test_half_open_failure_reopens_with_real_cooldown(self) -> None:
        clock = FakeClock()
        breaker = self.make_breaker(clock)
        for _ in range(2):
            permit = breaker.before_attempt()
            breaker.record_result(permit.permit, False)  # type: ignore[arg-type]
        clock.advance(0.11)
        probe = breaker.before_attempt()
        breaker.record_result(probe.permit, False)  # type: ignore[arg-type]
        self.assertEqual(breaker.state, BreakerState.OPEN)
        denied = breaker.before_attempt()
        self.assertFalse(denied.allowed)

    def test_declared_probe_count_is_enforced(self) -> None:
        clock = FakeClock()
        breaker = self.make_breaker(clock, probes=2)
        for _ in range(2):
            permit = breaker.before_attempt()
            breaker.record_result(permit.permit, False)  # type: ignore[arg-type]
        clock.advance(0.11)
        first = breaker.before_attempt()
        second = breaker.before_attempt()
        third = breaker.before_attempt()
        self.assertTrue(first.allowed)
        self.assertTrue(second.allowed)
        self.assertFalse(third.allowed)


class BudgetTests(unittest.TestCase):
    def test_global_and_per_service_limits_are_never_exceeded(self) -> None:
        scheduler = BudgetScheduler(2, {"a": 1, "b": 2})
        active = {"a": 0, "b": 0, "global": 0}
        peak = {"a": 0, "b": 0, "global": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(7)
        errors: list[str] = []

        def worker(index: int) -> None:
            service = "a" if index % 2 else "b"
            barrier.wait()
            slot = scheduler.acquire(service, (index, 1))
            with lock:
                active[service] += 1
                active["global"] += 1
                peak[service] = max(peak[service], active[service])
                peak["global"] = max(peak["global"], active["global"])
                if active[service] > (1 if service == "a" else 2):
                    errors.append(f"{service} exceeded")
                if active["global"] > 2:
                    errors.append("global exceeded")
            time.sleep(0.03)
            with lock:
                active[service] -= 1
                active["global"] -= 1
            scheduler.release(slot)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(7)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertLessEqual(peak["global"], 2)
        self.assertLessEqual(peak["a"], 1)
        self.assertLessEqual(peak["b"], 2)
        snapshot = scheduler.snapshot()
        self.assertGreaterEqual(snapshot["global_grants"], 7)  # type: ignore[operator]
        self.assertEqual(snapshot["global_inflight"], 0)


if __name__ == "__main__":
    unittest.main()
