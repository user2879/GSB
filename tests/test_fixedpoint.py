from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal

from fixedseason.attribution import attribute_stage_errors
from fixedseason.basis import build_basis
from fixedseason.engine import FitEngine
from fixedseason.fixedpoint import NormalEquationResult, solve_fixed_point
from fixedseason.interpolation import interpolate_series, round_decimal_half_away
from fixedseason.jsonutil import fingerprint
from fixedseason.model import FitConfig, SeriesRecord


class InterpolationBasisTests(unittest.TestCase):
    def test_window_only_linear_and_edge_imputation(self) -> None:
        series = SeriesRecord(
            "s", (0.0, 1.0, 2.0, 3.0, 4.0), (1.0, None, None, 4.0, None)
        )
        result = interpolate_series(series)
        self.assertEqual(result.values, (1.0, 2.0, 3.0, 4.0, 4.0))
        self.assertEqual(result.metadata["observed_indices"], [0, 3])
        self.assertEqual(result.metadata["tie_breaks"]["nearest_distance_equal"], "lower_index_left")
        self.assertTrue(result.metadata["uses_only_observed_in_window"])
        self.assertEqual(result.metadata["imputed"][-1]["rule"], "edge_nearest")

    def test_half_away_rounding_rule(self) -> None:
        self.assertEqual(round_decimal_half_away(Decimal("1.5")), Decimal("2"))
        self.assertEqual(round_decimal_half_away(Decimal("-1.5")), Decimal("-2"))

    def test_basis_metadata_and_pure_trend_degeneration(self) -> None:
        config = FitConfig(harmonic_order=0)
        basis = build_basis(8, config)
        self.assertTrue(basis.degenerate_pure_trend)
        self.assertEqual(len(basis.columns), 2)
        self.assertEqual([column["phase"] for column in basis.columns], ["constant", "linear"])
        harmonic = build_basis(8, FitConfig(harmonic_order=1))
        self.assertEqual(harmonic.columns[2]["harmonic_order"], 1)
        self.assertEqual(harmonic.columns[2]["phase"], "cos")
        self.assertEqual(harmonic.columns[3]["phase"], "sin")


class FixedPointTests(unittest.TestCase):
    def test_deterministic_pivot_fallback_trace(self) -> None:
        matrix = (
            (100, 0, 1, 0),
            (0, 1, 0, 1),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
        )
        normal = NormalEquationResult(
            matrix=matrix,
            rhs=(10, 2, 3, 4),
            raw_matrix=matrix,
            raw_rhs=(10, 2, 3, 4),
            regularization_integer=0,
            accumulator_max_bits=7,
            downshift=0,
            accumulation_order=("time_index_ascending", "column_index_ascending"),
            artifact_fingerprint=fingerprint(matrix),
        )
        config = FitConfig(
            harmonic_order=1,
            fixed_point_bits=48,
            quantization_shift=12,
            pivot_threshold=2,
        )
        first = solve_fixed_point(normal, config)
        second = solve_fixed_point(normal, config)
        self.assertEqual(first, second)
        self.assertEqual(first.effective_dimension, 1)
        self.assertEqual(first.effective_harmonic_order, 0)
        self.assertTrue(first.trend_disabled)
        self.assertEqual(len(first.reductions), 2)
        self.assertEqual(first.reductions[0]["reason"], "PIVOT_BELOW_THRESHOLD")
        self.assertEqual(first.reductions[1]["rule"], "disable_trend_and_resolve_intercept_only")
        self.assertTrue(first.pivots)
        self.assertIn("candidates", first.pivots[0])
        self.assertIn(
            "row_operation_trace_before_failure", first.reductions[0]
        )
        self.assertIn(
            "failed_pivot_candidates", first.reductions[0]
        )
        failed_operations = first.reductions[0]["row_operation_trace_before_failure"]
        self.assertTrue(failed_operations)
        self.assertIn("intermediate_remainder", failed_operations[0])
        self.assertIn("rounding_direction", failed_operations[0])
        self.assertEqual(first.coefficients, (0,))

    def test_repeated_engine_runs_are_byte_stable(self) -> None:
        series = [
            SeriesRecord(
                "a",
                tuple(float(i) for i in range(24)),
                tuple(100 + 2.0 * i + (3 if i % 4 == 0 else -1) for i in range(24)),
            )
        ]
        config = FitConfig(harmonic_order=2, quantization_shift=12, version=1)
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            report_one = FitEngine(one, config, series).run(version=1, force_rerun=True)
            report_two = FitEngine(two, config, series).run(version=1, force_rerun=True)
        self.assertEqual(report_one["series"], report_two["series"])
        self.assertEqual(
            report_one["series"][0]["coefficient_values"],
            report_two["series"][0]["coefficient_values"],
        )
        self.assertFalse(
            report_one["series"][0]["basis"]["degenerate_pure_trend"]
        )
        ordinary = FitEngine(
            tempfile.mkdtemp(), config.with_changes(harmonic_order=0), series
        ).run(version=1, force_rerun=True)
        self.assertTrue(ordinary["series"][0]["basis"]["degenerate_pure_trend"])
        self.assertTrue(
            ordinary["series"][0]["model_degenerate_flags"][
                "configured_harmonic_zero_pure_trend"
            ]
        )


class AttributionTests(unittest.TestCase):
    def test_zero_truth_uses_absolute_branch_and_first_stage(self) -> None:
        result = attribute_stage_errors(
            names=("zero", "nonzero"),
            truth=(0.0, 2.0),
            interpolated=(0.001, 2.0),
            normal_accumulated=(0.001, 2.0),
            eliminated=(0.001, 2.0),
            final=(0.002, 2.01),
            relative_tolerance=0.001,
            absolute_tolerance=0.0005,
        )
        zero_entry = result["stages"]["interpolation"]["coefficients"][0]
        self.assertEqual(zero_entry["metric_kind"], "absolute_zero_truth")
        self.assertTrue(zero_entry["exceeded"])
        self.assertEqual(result["first_exceeded_stage"], "interpolation")
        self.assertIn("zero", result["zero_truth_branch_coefficients"])
        final_nonzero = result["stages"]["coefficient_back_substitution"]["coefficients"][1]
        self.assertEqual(final_nonzero["metric_kind"], "relative_nonzero_truth")


if __name__ == "__main__":
    unittest.main()
