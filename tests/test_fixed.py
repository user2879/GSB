from __future__ import annotations

import unittest

from streamagg.fixed import (
    INT64_MAX,
    INT64_MIN,
    build_error_attribution,
    clamp_i64,
    quantize,
    trunc_div_toward_zero,
)


class FixedPointTests(unittest.TestCase):
    def test_truncation_is_toward_zero(self) -> None:
        self.assertEqual(trunc_div_toward_zero(7, 2), 3)
        self.assertEqual(trunc_div_toward_zero(-7, 2), -3)
        self.assertEqual(trunc_div_toward_zero(9, 3), 3)

    def test_quantize_exact_for_integer_metric_and_integer_scale(self) -> None:
        self.assertEqual(quantize(-37, 1000), (-37000, 0))

    def test_quantize_saturates_signed_int64(self) -> None:
        value, error = quantize(INT64_MAX, 2)
        self.assertEqual(value, INT64_MAX)
        self.assertEqual(error, INT64_MAX)
        self.assertEqual(clamp_i64(INT64_MIN - 1), INT64_MIN)

    def test_attribution_three_segments_and_mean_bound(self) -> None:
        attribution, sum_fixed, mean_fixed = build_error_attribution(
            count=4,
            sum_exact=101,
            quantization_error_sum=2,
            max_single_quantization_error=2,
            late_accepted_abs_amount=8,
            boundary_abs_amount=3,
            threshold_bp=6000,
        )
        self.assertEqual(sum_fixed, 101)
        self.assertEqual(mean_fixed, 25)
        self.assertEqual(attribution.quantization_truncation, 2)
        self.assertEqual(attribution.late_admission, 8)
        self.assertEqual(attribution.window_boundary, 3)
        self.assertEqual(attribution.sum_bound, 13)
        self.assertEqual(attribution.mean_bound, 5)
        self.assertEqual(attribution.dominant_segment, "LATE_ADMISSION")
        self.assertEqual(attribution.dominant_share_bp, 6154)

    def test_dominant_requires_threshold(self) -> None:
        attribution, _, _ = build_error_attribution(
            count=2,
            sum_exact=0,
            quantization_error_sum=0,
            max_single_quantization_error=0,
            late_accepted_abs_amount=1,
            boundary_abs_amount=1,
            threshold_bp=6000,
        )
        self.assertEqual(attribution.dominant_segment, "NONE")
        self.assertEqual(attribution.dominant_share_bp, 5000)

    def test_all_zero_error_is_stable(self) -> None:
        attribution, _, _ = build_error_attribution(
            count=3,
            sum_exact=9,
            quantization_error_sum=0,
            max_single_quantization_error=0,
            late_accepted_abs_amount=0,
            boundary_abs_amount=0,
            threshold_bp=6000,
        )
        self.assertEqual(attribution.dominant_segment, "NONE")
        self.assertEqual(attribution.dominant_share_bp, 0)
        self.assertEqual(attribution.mean_bound, 1)


if __name__ == "__main__":
    unittest.main()
