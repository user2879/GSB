from __future__ import annotations

import unittest

from tests.helpers import SRC  # noqa: F401

from dbml.errors import MigrationError
from dbml.expression import EvalValue, parse_expression
from dbml.fixedpoint import QuantizationSpec, quantize_decimal


class FixedPointTests(unittest.TestCase):
    def test_single_value_quantization_and_directions(self) -> None:
        toward_zero = quantize_decimal("1.239", QuantizationSpec(2, "toward_zero"))
        floor = quantize_decimal("-1.239", QuantizationSpec(2, "floor"))
        ceil = quantize_decimal("-1.239", QuantizationSpec(2, "ceil"))
        half_even = quantize_decimal("1.245", QuantizationSpec(2, "nearest_half_even"))
        self.assertEqual((toward_zero.units, toward_zero.error_units), (123, 1))
        self.assertEqual(floor.units, -124)
        self.assertEqual(ceil.units, -123)
        self.assertEqual(half_even.units, 124)

    def test_saturation_reject_and_clamp(self) -> None:
        reject = QuantizationSpec(0, saturation="reject", max_units=10)
        with self.assertRaises(MigrationError) as raised:
            quantize_decimal("11", reject)
        self.assertEqual(raised.exception.code, "E_MIGRATE")
        clamp = QuantizationSpec(0, saturation="clamp", max_units=10)
        value = quantize_decimal("11", clamp)
        self.assertEqual(value.units, 10)
        self.assertTrue(value.saturated)
        self.assertEqual(value.error_units, 1)

    def test_expression_uses_integer_grid_and_sorted_paths(self) -> None:
        expression = parse_expression("(a + b) / 2", step_id="expr")
        values = {
            "a": EvalValue(
                nominal=100,
                scale=2,
                lo=99,
                hi=101,
                terms={("source", "a"): 1},
            ),
            "b": EvalValue(
                nominal=50,
                scale=2,
                lo=50,
                hi=50,
                terms={},
            ),
        }
        result = expression.evaluate(
            lambda column: values[column], target_scale=2, rounding="toward_zero"
        )
        self.assertEqual((result.units, result.scale), (75, 2))
        self.assertGreater(result.error_bound, 0)
        paths = [tuple(term["path"]) for term in result.terms]
        self.assertEqual(paths, sorted(paths))
        self.assertIn(("source", "a"), paths)

    def test_expression_multiplication_keeps_integer_units(self) -> None:
        expression = parse_expression("a * b", step_id="expr")
        values = {
            "a": EvalValue(nominal=123, scale=2, lo=123, hi=123, terms={}),
            "b": EvalValue(nominal=25, scale=1, lo=25, hi=25, terms={}),
        }
        result = expression.evaluate(
            lambda column: values[column], target_scale=2, rounding="toward_zero"
        )
        self.assertEqual(result.units, 307)
        self.assertEqual(result.error_bound, 1)


if __name__ == "__main__":
    unittest.main()

