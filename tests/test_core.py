"""Unit tests for pure domain components (no sockets required)."""

import os
import unittest

from meshbench.breaker import CLOSED, HALF_OPEN, OPEN, Breaker
from meshbench.budget import (SCALE, SLOW, SUCCESS, FAILURE, TIMEOUT,
                              WindowCounters, compute_receipt)
from meshbench.config import (BUCKET_SPAN, RuleError, load_config,
                              parse_config)
from meshbench.hashing import choose, stable_bucket
from meshbench.canonical import canonical_dumps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V1 = os.path.join(ROOT, "samples", "rules_v1.json")


class ConfigValidationTests(unittest.TestCase):
    def test_valid_sample_loads(self):
        config = load_config(V1)
        self.assertEqual(config.version, 1)
        self.assertIn("orders", config.services)

    def _base(self):
        import json
        return json.load(open(V1, encoding="utf-8"))

    def test_reject_non_loopback(self):
        doc = self._base()
        doc["services"][0]["instances"][0]["host"] = "10.0.0.1"
        with self.assertRaises(RuleError) as ctx:
            parse_config(doc)
        self.assertEqual(ctx.exception.code, "E_RULE")

    def test_reject_duplicate_port(self):
        doc = self._base()
        doc["services"][1]["instances"][0]["port"] = 19001
        with self.assertRaises(RuleError):
            parse_config(doc)

    def test_reject_zero_total_weight(self):
        doc = self._base()
        for inst in doc["services"][0]["instances"]:
            inst["weight"] = 0
        with self.assertRaises(RuleError):
            parse_config(doc)

    def test_reject_budget_range_and_nonpositive_window(self):
        doc = self._base()
        doc["routes"][0]["budget_permyriad"] = 10001
        with self.assertRaises(RuleError):
            parse_config(doc)
        doc = self._base()
        doc["routes"][0]["window_minutes"] = 0
        with self.assertRaises(RuleError):
            parse_config(doc)

    def test_duplicate_service_id_rejected(self):
        doc = self._base()
        doc["services"][1]["service_id"] = doc["services"][0]["service_id"]
        with self.assertRaises(RuleError):
            parse_config(doc)


class HashingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(V1)
        self.service = self.config.services["orders"]

    def test_bucket_bounds(self):
        for i in range(200):
            bucket = stable_bucket(1, "orders-api", f"client-{i}")
            self.assertIn(bucket, range(BUCKET_SPAN))

    def test_deterministic_and_order_independent(self):
        instances_a = self.service.instances
        instances_b = tuple(reversed(self.service.instances))
        for i in range(50):
            key = f"client-{i}"
            self.assertEqual(
                choose(1, "orders-api", key, instances_a).instance_id,
                choose(1, "orders-api", key, instances_b).instance_id,
            )

    def test_client_affinity(self):
        chosen = {
            choose(1, "orders-api", "affinity-key", self.service.instances).instance_id
            for _ in range(20)
        }
        self.assertEqual(len(chosen), 1)

    def test_zero_weight_never_chosen(self):
        for i in range(300):
            inst = choose(1, "orders-api", f"c{i}", self.service.instances)
            self.assertNotEqual(inst.weight, 0)
            self.assertNotEqual(inst.instance_id, "orders-c")

    def test_weight_distribution_approximately_proportional(self):
        # Floor-scaled integer buckets introduce bounded discretization, so use
        # a large deterministic sample and a tolerance above that bound.
        counts = {}
        for i in range(20000):
            inst = choose(1, "orders-api", f"dist-key-{i}", self.service.instances)
            counts[inst.instance_id] = counts.get(inst.instance_id, 0) + 1
        total = sum(counts.values())
        self.assertAlmostEqual(counts["orders-a"] / total, 0.6, delta=0.02)
        self.assertAlmostEqual(counts["orders-b"] / total, 0.4, delta=0.02)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(V1)
        self.route = self.config.routes["orders-api"]

    def test_receipt_order_independent_and_byte_stable(self):
        counters = WindowCounters(success=7, failure=2, timeout=1, slow=2)
        r1 = compute_receipt(self.route, 1, 3, counters)
        # Same totals, different "arrival order" cannot change content.
        rebuilt = WindowCounters().add(FAILURE).add(TIMEOUT).add(SLOW).add(SLOW)
        for _ in range(7):
            rebuilt = rebuilt.add(SUCCESS)
        for _ in range(1):
            rebuilt = rebuilt.add(FAILURE)
        r2 = compute_receipt(self.route, 1, 3, rebuilt)
        self.assertEqual(r1.canonical_bytes, r2.canonical_bytes)
        self.assertEqual(r1.digest, r2.digest)

    def test_fixed_point_math_and_error_bound(self):
        counters = WindowCounters(success=1, failure=1, timeout=0, slow=0)
        receipt = compute_receipt(self.route, 1, 0, counters)
        payload = receipt.payload
        self.assertEqual(payload["scale"], SCALE)
        self.assertEqual(payload["truncation"], "floor")
        # 1/2 availability -> exactly 500000 micro-units.
        self.assertEqual(payload["availability_sli_scaled"], SCALE // 2)
        self.assertEqual(payload["quant_error_max"]["denominator"], SCALE)

    def test_insufficient_sample_blocks_exhaustion(self):
        # min_samples is 5; only 2 samples -> insufficient, never exhausted.
        counters = WindowCounters(success=0, failure=2, timeout=0, slow=0)
        payload = compute_receipt(self.route, 1, 0, counters).payload
        self.assertTrue(payload["insufficient_sample"])
        self.assertFalse(payload["budget_exhausted"])

    def test_budget_exhaustion_only_when_enough_samples(self):
        # budget 20% of 10 = 2 allowed; 4 bad events exceeds it.
        counters = WindowCounters(success=6, failure=4, timeout=0, slow=0)
        payload = compute_receipt(self.route, 1, 0, counters).payload
        self.assertFalse(payload["insufficient_sample"])
        self.assertTrue(payload["budget_exhausted"])
        self.assertLess(payload["remaining_scaled"], 0)

    def test_canonical_bytes_is_canonical(self):
        counters = WindowCounters(success=3, failure=1, timeout=1, slow=1)
        receipt = compute_receipt(self.route, 1, 0, counters)
        self.assertEqual(receipt.canonical_bytes,
                         canonical_dumps(receipt.payload))


class BreakerTests(unittest.TestCase):
    def test_opens_after_threshold_and_fast_fails(self):
        breaker = Breaker("inst", failure_threshold=3, cooldown_ms=10_000,
                          half_open_probes=1)
        events = [breaker.record_failure() for _ in range(3)]
        self.assertIsNone(events[0])
        self.assertIsNone(events[1])
        self.assertEqual(events[2]["to_state"], OPEN)
        allowed, state = breaker.allow_request()
        self.assertFalse(allowed)
        self.assertEqual(state, OPEN)

    def test_half_open_success_closes(self):
        breaker = Breaker("inst", 1, cooldown_ms=5, half_open_probes=1)
        breaker.record_failure()
        import time
        time.sleep(0.02)
        allowed, state = breaker.allow_request()
        self.assertTrue(allowed)
        self.assertEqual(state, HALF_OPEN)
        migration = breaker.record_success()
        self.assertEqual(migration["to_state"], CLOSED)

    def test_half_open_failure_reopens_and_resets_counter(self):
        breaker = Breaker("inst", 2, cooldown_ms=5, half_open_probes=1)
        breaker.record_failure()
        breaker.record_failure()
        import time
        time.sleep(0.02)
        breaker.allow_request()  # enter HALF_OPEN
        migration = breaker.record_failure()
        self.assertEqual(migration["to_state"], OPEN)
        self.assertEqual(migration["reason"], "half_open_probe_failed")
        self.assertEqual(breaker.consecutive_failures, 0)


if __name__ == "__main__":
    unittest.main()
