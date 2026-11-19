from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bucketpack.canonical import dumps
from bucketpack.engine import deterministic_view, run_full
from bucketpack.numeric import INT64_MAX, error_bound, quantize

from .common import sample_spec, write_known_data_spec


class NumericExpansionTests(unittest.TestCase):
    def test_quantize_truncates_toward_zero(self) -> None:
        self.assertEqual(quantize(1.239, 2).value, 123)
        self.assertEqual(quantize(-1.239, 2).value, -123)
        self.assertEqual(quantize(1.999, 0).value, 1)
        self.assertTrue(quantize(10.0**30, 9).saturated)
        self.assertEqual(quantize(10.0**30, 9).value, INT64_MAX)

    def test_quantization_error_bound(self) -> None:
        q = quantize(0.129, 2)
        self.assertEqual(q.value, 12)
        self.assertEqual(str(q.error), "9/1000")
        self.assertEqual(error_bound(2, q.error, False).numerator, 1)
        self.assertEqual(error_bound(2, q.error, False).denominator, 100)

    def test_known_bucket_expansion_and_downstream_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            spec = write_known_data_spec(Path(td))
            state = run_full(spec)
        buckets = state.expansions["known"].buckets
        # q values 0,999,1000,1999; equal boundaries enter the right bucket.
        self.assertEqual(buckets["known.vb"].values, (1, 1, 2, 3))
        # downstream bucketing of bucket ids 1,1,2,3 -> bisect_right([1,2]).
        self.assertEqual(buckets["known.vb2"].values, (1, 1, 2, 2))
        self.assertEqual(buckets["known.vb"].counts, (0, 2, 1, 1))
        self.assertEqual(buckets["known.vb"].width_bits, 2)
        self.assertEqual(buckets["known.vb"].cell_width, 3)

    def test_repeated_runs_are_byte_identical(self) -> None:
        spec = sample_spec()
        first = run_full(spec)
        second = run_full(spec)
        self.assertEqual(dumps(deterministic_view(first.result)), dumps(deterministic_view(second.result)))
        self.assertEqual(first.result["content_digest"], second.result["content_digest"])
        self.assertEqual(
            dumps(first.expansions["orders"].to_dict()),
            dumps(second.expansions["orders"].to_dict()),
        )

    def test_null_fill_does_not_change_row_width(self) -> None:
        spec = sample_spec()
        state = run_full(spec)
        for expansion in state.expansions.values():
            self.assertEqual(
                expansion.row_width,
                sum(bucket.cell_width for bucket in expansion.buckets.values()),
            )
            self.assertGreater(expansion.row_width, 0)


if __name__ == "__main__":
    unittest.main()
