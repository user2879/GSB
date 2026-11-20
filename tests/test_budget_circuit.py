from __future__ import annotations

import threading
import time
import unittest

from meshgov.budget import ConcurrencyGate
from meshgov.circuit import CLOSED, HALF_OPEN, OPEN, CircuitManager
from tests.helpers import instance, make_rules


class BudgetTests(unittest.TestCase):
    def test_global_and_service_limits_are_never_exceeded(self) -> None:
        gate = ConcurrencyGate(4, {"alpha": 2, "beta": 3})
        lock = threading.Lock()
        active_global = 0
        active_services = {"alpha": 0, "beta": 0}
        peaks = {"global": 0, "alpha": 0, "beta": 0}

        def worker(service: str) -> None:
            nonlocal active_global
            gate.acquire(service)
            try:
                with lock:
                    active_global += 1
                    active_services[service] += 1
                    peaks["global"] = max(peaks["global"], active_global)
                    peaks[service] = max(peaks[service], active_services[service])
                time.sleep(0.01)
                with lock:
                    active_global -= 1
                    active_services[service] -= 1
            finally:
                gate.release(service)

        threads = [
            threading.Thread(target=worker, args=("alpha" if index % 2 else "beta",))
            for index in range(28)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        snapshot = gate.snapshot()
        gate.close()
        self.assertLessEqual(peaks["global"], 4)
        self.assertLessEqual(peaks["alpha"], 2)
        self.assertLessEqual(peaks["beta"], 3)
        self.assertEqual(snapshot.global_active, 0)
        self.assertEqual(snapshot.global_peak, peaks["global"])


class CircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        rules = make_rules()
        self.instance = rules.service("alpha").instances[0]
        self.events: list[dict] = []
        self.circuits = CircuitManager(
            [self.instance],
            on_event=self.events.append,
        )

    def test_open_half_open_probe_and_close(self) -> None:
        now = 1000.0
        self.assertTrue(self.circuits.admit(self.instance, now).allowed)
        self.circuits.record_failure(self.instance, now + 1)
        self.circuits.record_failure(self.instance, now + 2)
        self.assertEqual(self.circuits.states()[0].state, OPEN)
        self.assertFalse(self.circuits.admit(self.instance, now + 50).allowed)
        probe = self.circuits.admit(self.instance, now + 200)
        self.assertTrue(probe.allowed)
        self.assertTrue(probe.probe)
        self.assertFalse(self.circuits.admit(self.instance, now + 201).allowed)
        self.circuits.record_success(self.instance, now + 202, probe=True)
        self.assertEqual(self.circuits.states()[0].state, CLOSED)
        self.assertEqual([event["event"] for event in self.events], ["opened", "half_open", "closed"])

    def test_failed_probe_reopens_with_new_cooldown(self) -> None:
        now = 1000.0
        self.circuits.record_failure(self.instance, now)
        self.circuits.record_failure(self.instance, now + 1)
        self.assertTrue(self.circuits.admit(self.instance, now + 101).allowed)
        self.circuits.record_failure(self.instance, now + 102, probe=True)
        state = self.circuits.states()[0]
        self.assertEqual(state.state, OPEN)
        self.assertGreater(state.cooldown_until_mono_ms, now + 102)
        self.assertEqual(self.events[-1]["event"], "reopened")

    def test_late_inflight_success_does_not_bypass_open_cooldown(self) -> None:
        now = 1000.0
        self.circuits.record_failure(self.instance, now)
        self.circuits.record_failure(self.instance, now + 1)
        self.assertEqual(self.circuits.states()[0].state, OPEN)
        self.circuits.record_success(self.instance, now + 2)
        self.assertEqual(self.circuits.states()[0].state, OPEN)
        self.assertFalse(self.circuits.admit(self.instance, now + 3).allowed)

    def test_late_nonprobe_success_does_not_close_half_open(self) -> None:
        now = 1000.0
        self.circuits.record_failure(self.instance, now)
        self.circuits.record_failure(self.instance, now + 1)
        probe = self.circuits.admit(self.instance, now + 101)
        self.assertTrue(probe.probe)
        self.circuits.record_success(self.instance, now + 102, probe=False)
        self.assertEqual(self.circuits.states()[0].state, HALF_OPEN)
        self.circuits.record_success(self.instance, now + 103, probe=True)
        self.assertEqual(self.circuits.states()[0].state, CLOSED)


if __name__ == "__main__":
    unittest.main()
