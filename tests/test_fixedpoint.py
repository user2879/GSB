from __future__ import annotations

import unittest

from lateness_ledger.canonical import INT64_MAX, INT64_MIN, sat_add, trunc_div
from lateness_ledger.fixedpoint import (
    FixedEventValue,
    ScaleContext,
    aggregate_fixed,
    error_attribution,
)


class FixedPointTests(unittest.TestCase):
    def test_quantization_truncates_toward_zero(self) -> None:
        context = ScaleContext(3)
        self.assertEqual(context.quantize_decimal("1.23456"), 1234)
        self.assertEqual(context.quantize_decimal("-1.23456"), -1234)
        self.assertEqual(context.quantize_raw_integer(12345, 4), 1234)
        self.assertEqual(context.quantize_raw_integer(123, 4), 12)
        self.assertEqual(context.quantize_raw_integer(-123, 4), -12)

    def test_int64_saturation_and_truncation(self) -> None:
        self.assertEqual(sat_add(INT64_MAX, 1), INT64_MAX)
        self.assertEqual(sat_add(INT64_MIN, -1), INT64_MIN)
        self.assertEqual(trunc_div(-9, 2), -4)
        self.assertEqual(trunc_div(9, 2), 4)
        self.assertEqual(trunc_div(9, 0), 0)

    def test_aggregation_is_all_integer(self) -> None:
        rows = [
            FixedEventValue(0, 0, 1000, 2000),
            FixedEventValue(1, 0, 2000, 4000),
            FixedEventValue(2, 0, -3, 1000),
        ]
        stats = aggregate_fixed(rows)
        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.sum_fixed, 2997)
        self.assertEqual(stats.mean_fixed, 999)
        self.assertEqual(stats.weight_sum_fixed, 7000)
        self.assertEqual(stats.weighted_mean_fixed, trunc_div(1000 * 2000 + 2000 * 4000 - 3 * 1000, 7000))

    def test_three_segment_error_envelope_and_dominant_threshold(self) -> None:
        included = [
            FixedEventValue(0, 900, 500, 1),
            FixedEventValue(1, 1200, 4000, 1),
        ]
        accepted_late = [included[0]]
        errors = error_attribution(
            included=included,
            accepted_late=accepted_late,
            nominal_end_ms=1000,
            scale=3,
        )
        self.assertEqual(errors["single_value_error_bound_fixed"], 1)
        self.assertEqual(errors["segments_fixed"]["quantization_truncation"], 2)
        self.assertEqual(errors["segments_fixed"]["late_admission"], 500)
        self.assertEqual(errors["segments_fixed"]["boundary_attribution"], 4000)
        self.assertEqual(errors["aggregation_error_bound_fixed"], 4502)
        self.assertEqual(errors["mean_error_bound_fixed"], 2252)
        self.assertEqual(errors["dominant_segment"], "boundary_attribution")
        self.assertEqual(errors["dominant_threshold_ratio"], "1/2")
        self.assertEqual(errors["truncation_direction"], "toward_zero")
        self.assertEqual(errors["overflow_rule"], "saturate_int64")


if __name__ == "__main__":
    unittest.main()
