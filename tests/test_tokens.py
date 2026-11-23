"""Fixed-point token bucket tests."""

import unittest

from workbench.tokens import TokenBucket


class TokenBucketTest(unittest.TestCase):
    def test_starts_full_and_consumes(self):
        bucket = TokenBucket(capacity_tokens=3, refill_interval_ms=50,
                             cost_tokens=1, scale=3)
        d = bucket.evaluate(0)
        self.assertEqual((d.before, d.refilled, d.allowed, d.after),
                         (3000, 0, True, 2000))

    def test_integer_refill_and_clamp(self):
        bucket = TokenBucket(capacity_tokens=2, refill_interval_ms=100,
                             cost_tokens=1, scale=2)
        bucket.evaluate(0)
        d = bucket.evaluate(450)
        self.assertEqual((d.before, d.refilled, d.allowed, d.after),
                         (100, 100, True, 100))

    def test_denied_when_insufficient(self):
        bucket = TokenBucket(capacity_tokens=1, refill_interval_ms=1000,
                             cost_tokens=1, scale=0)
        self.assertTrue(bucket.evaluate(0).allowed)
        d = bucket.evaluate(10)
        self.assertFalse(d.allowed)
        self.assertEqual(d.after, 0)

    def test_partial_interval_no_refill(self):
        bucket = TokenBucket(capacity_tokens=1, refill_interval_ms=100,
                             cost_tokens=1, scale=1)
        bucket.evaluate(0)
        d = bucket.evaluate(99)
        self.assertEqual(d.refilled, 0)
        self.assertFalse(d.allowed)

    def test_scale_quantization(self):
        for scale in range(0, 7):
            bucket = TokenBucket(2, 50, 1, scale)
            d = bucket.evaluate(0)
            self.assertEqual(d.before, 2 * 10 ** scale)
            self.assertEqual(d.after, 1 * 10 ** scale)

    def test_trajectory_deterministic(self):
        arrivals = [0, 10, 20, 30, 200, 210, 1000]
        traces = []
        for _ in range(2):
            bucket = TokenBucket(3, 50, 1, 3)
            traces.append([(d.before, d.refilled, d.allowed, d.after)
                           for d in (bucket.evaluate(a)
                                     for a in arrivals)])
        self.assertEqual(traces[0], traces[1])


if __name__ == "__main__":
    unittest.main()
