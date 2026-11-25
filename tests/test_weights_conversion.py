import json
import os
import tempfile
import unittest

from fxpdiag.convert_service import full_roundtrip
from fxpdiag.conversion import v2_to_v1_row, v1_to_v2_with_sidecar
from fxpdiag.errors import FailureList
from fxpdiag.models import V1WeightRow, V2WeightRow, Scheme
from fxpdiag.weights import (
    V1_LINE_WIDTH,
    format_v1_line,
    parse_v1_line,
    write_v2,
    read_v2,
)


class WeightFormatTests(unittest.TestCase):
    def test_v1_fixed_width_roundtrip(self):
        row = V1WeightRow("attention", 7, 1234, -56, 16)
        line = format_v1_line(row)
        self.assertEqual(len(line), V1_LINE_WIDTH)
        failures = FailureList()
        parsed = parse_v1_line(line, "mem", failures)
        self.assertEqual(len(failures), 0)
        self.assertEqual(parsed, row)

    def test_v1_range_validation(self):
        failures = FailureList()
        bad = V1WeightRow("l", 0, 10_000_000, 0, 8)
        from fxpdiag.weights import validate_v1_row
        self.assertFalse(validate_v1_row(bad, "mem", failures))
        self.assertEqual(failures.codes(), ["E_FIELD_RANGE"])

    def test_v1_bitwidth_enum_validation(self):
        failures = FailureList()
        bad = V1WeightRow("l", 0, 10, 0, 13)
        from fxpdiag.weights import validate_v1_row
        self.assertFalse(validate_v1_row(bad, "mem", failures))
        self.assertEqual(failures.codes(), ["E_BITWIDTH_ENUM"])

    def test_v2_non_finite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.json")
            write_v2(path, [V2WeightRow("l", 0, float("nan"),
                                        Scheme.Q8_8_NORM.value, "HALF_EVEN")])
            failures = FailureList()
            rows = read_v2(path, failures)
            self.assertEqual(rows, [])
            self.assertEqual(failures.codes(), ["E_NON_FINITE"])

    def test_roundtrip_sidecar_restores_original(self):
        v2_rows = [
            V2WeightRow("l", 0, 1.2347, Scheme.Q8_8_NORM.value, "HALF_UP"),
            V2WeightRow("l", 1, 0.0058, Scheme.Q4_3_NORM_NO_BIAS.value, "TRUNC"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = full_roundtrip(v2_rows, tmp)
            self.assertTrue(os.path.exists(result["v1_path"]))
            self.assertTrue(
                os.path.exists(result["v1_path"] + ".sidecar.json")
            )
            with open(result["restored_v2_path"]) as handle:
                restored = json.load(handle)
            self.assertAlmostEqual(restored["rows"][0]["scale"], 1.2347, places=12)
            self.assertEqual(restored["rows"][0]["rounding_mode"], "HALF_UP")
            self.assertGreater(
                sum(result["layer_roundtrip_loss"].values()), 0.0
            )

    def test_v2_to_v1_residual_recorded(self):
        v1_row, sidecar = v2_to_v1_row(
            V2WeightRow("l", 0, 1.2349, Scheme.Q16_16_NORM.value, "FLOOR")
        )
        self.assertEqual(v1_row.scale_permille, 1235)
        self.assertNotAlmostEqual(sidecar.scale_residual, 0.0)
        restored, record = v1_to_v2_with_sidecar(v1_row, sidecar)
        self.assertTrue(record.sidecar_used)
        self.assertAlmostEqual(restored.scale, 1.2349, places=12)
        self.assertEqual(restored.rounding_mode, "FLOOR")


if __name__ == "__main__":
    unittest.main()
