from __future__ import annotations

import tempfile
import unittest

from fixedseason.diagnostics import (
    classify_sign_origin,
    compare_two_widths,
    run_width_sign_check,
)
from fixedseason.model import FitConfig, SeriesRecord


def artifact(slope: float, *, elim_slope: float | None = None, reductions: int = 0) -> dict:
    elim = slope if elim_slope is None else elim_slope
    stages = ["interpolation", "normal_equation_accumulation", "integer_elimination", "coefficient_back_substitution"]
    values = [1.0, slope, slope, elim, slope]
    out = {}
    for index, stage in enumerate(stages):
        out[stage] = {"cumulative_stage_value": [0.0, values[index + 1]]}
    return {
        "series_id": "s",
        "error_attribution": {
            "truth_coefficients": [0.0, slope],
            "stage_order": stages,
            "stages": out,
        },
        "elimination": {"reductions": [{"reason": "PIVOT_BELOW_THRESHOLD"}] * reductions},
    }


class SignDiagnosticsTests(unittest.TestCase):
    def test_classifies_pivot_and_rounding_origins(self) -> None:
        pivot = classify_sign_origin(artifact(1.0, elim_slope=-1.0))
        self.assertIn("elimination_pivot", pivot["causes"])
        rounding_artifact = artifact(1.0)
        rounding_artifact["error_attribution"]["stages"][
            "coefficient_back_substitution"
        ]["cumulative_stage_value"][1] = -1.0
        rounding = classify_sign_origin(rounding_artifact)
        self.assertIn("fixed_point_rounding", rounding["causes"])

    def test_cross_width_divergence_stage(self) -> None:
        positive = artifact(1.0)
        negative = artifact(1.0)
        negative["error_attribution"]["stages"][
            "normal_equation_accumulation"
        ]["cumulative_stage_value"][1] = -1.0
        for stage in (
            "integer_elimination",
            "coefficient_back_substitution",
        ):
            negative["error_attribution"]["stages"][stage][
                "cumulative_stage_value"
            ][1] = -1.0
        comparison = compare_two_widths(positive, negative)
        self.assertTrue(comparison["flipped"])
        self.assertEqual(comparison["divergence_stage"], "normal_equation_accumulation")
        self.assertEqual(comparison["attributed_cause"], "normal_equation_quantization")

    def test_width_check_integration_shape(self) -> None:
        series = [
            SeriesRecord(
                "a",
                tuple(float(i) for i in range(28)),
                tuple(50 + 0.01 * i + (2 if i % 3 == 0 else -1) for i in range(28)),
            )
        ]
        config = FitConfig(
            harmonic_order=1,
            fixed_point_bits=48,
            quantization_shift=12,
            thread_pool_workers=2,
        )
        with tempfile.TemporaryDirectory() as temp:
            report = run_width_sign_check(temp, series, config, [32, 48])
        self.assertEqual(report["widths"], [32, 48])
        self.assertEqual(len(report["series_comparisons"]), 1)
        self.assertIn("flipped", report["series_comparisons"][0])


if __name__ == "__main__":
    unittest.main()
