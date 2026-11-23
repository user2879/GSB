from __future__ import annotations

import unittest

from imv.fixed import AggregateAccumulator, FixedPointContext
from imv.util import INT64_MAX, INT64_MIN


class FixedPointTests(unittest.TestCase):
    def test_truncation_directions(self) -> None:
        toward_zero = FixedPointContext(100, "toward_zero", "saturate_int64")
        floor = FixedPointContext(100, "floor", "saturate_int64")
        ceil = FixedPointContext(100, "ceil", "saturate_int64")
        self.assertEqual(toward_zero.quantize(-1.999), -199)
        self.assertEqual(floor.quantize(-1.999), -200)
        self.assertEqual(ceil.quantize(-1.999), -199)
        self.assertEqual(toward_zero.quantize(1.999), 199)
        self.assertEqual(floor.quantize(1.999), 199)
        self.assertEqual(ceil.quantize(1.999), 200)

    def test_int64_saturation(self) -> None:
        fixed = FixedPointContext(1000, "toward_zero", "saturate_int64")
        self.assertEqual(fixed.quantize(10**20), INT64_MAX)
        self.assertEqual(fixed.quantize(-(10**20)), INT64_MIN)
        spec = {"func": "sum", "column": "amount", "alias": "sum_amount"}
        accumulator = AggregateAccumulator(spec, fixed)
        accumulator.add({"amount": 10**18})
        accumulator.add({"amount": 10**18})
        self.assertEqual(accumulator.value(), INT64_MAX)
        self.assertTrue(accumulator.saturated)

    def test_count_and_null_aggregate_semantics(self) -> None:
        fixed = FixedPointContext(100, "toward_zero", "saturate_int64")
        count = AggregateAccumulator({"func": "count", "column": "*", "alias": "rows"}, fixed)
        count.add({"amount": None})
        count.add({"amount": 2.0})
        self.assertEqual(count.value(), 200)
        amount_count = AggregateAccumulator({"func": "count", "column": "amount", "alias": "amounts"}, fixed)
        amount_count.add({"amount": None})
        amount_count.add({"amount": 2.0})
        self.assertEqual(amount_count.value(), 100)
        minimum = AggregateAccumulator({"func": "min", "column": "amount", "alias": "minimum"}, fixed)
        minimum.add({"amount": None})
        self.assertIsNone(minimum.value())
        total = AggregateAccumulator({"func": "sum", "column": "amount", "alias": "total"}, fixed)
        total.add({"amount": None})
        self.assertEqual(total.value(), 0)


if __name__ == "__main__":
    unittest.main()
