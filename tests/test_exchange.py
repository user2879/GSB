from __future__ import annotations

import copy
import unittest
from pathlib import Path

from quantcal.errors import QuantCalError
from quantcal.format_v1 import parse_v1, render_v1
from quantcal.format_v2 import (
    downgrade_v2_to_v1,
    parse_v2,
    upgrade_v1_to_v2,
)
from quantcal.util import load_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


class ExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v2_document = load_json(SAMPLE / "calibration_v2.json")
        cls.names = sorted(item["name"] for item in cls.v2_document["tensors"])
        cls.v1_text = (SAMPLE / "calibration_v1.txt").read_text(encoding="utf-8")

    def test_v1_fixed_width_round_trip_is_exact(self) -> None:
        v1 = parse_v1(self.v1_text, self.names)
        widths = v1.metadata["field_widths"]
        for line in self.v1_text.splitlines()[1:]:
            self.assertEqual(
                len(line),
                sum(widths.values()) + 3,
                "every v1 row must obey the header-declared fixed widths",
            )
        v2, losses = upgrade_v1_to_v2(v1)
        self.assertTrue(all(item.precision_tag == "v1_milli_exact_1e-3" for item in v2.tensors))
        self.assertTrue(all(row["approximated_as"] == "symmetric" for row in losses["losses"]))
        _, text, sidecar = downgrade_v2_to_v1(v2)
        reparsed = parse_v1(text, self.names)
        self.assertEqual(
            [item.name for item in v1.tensors],
            [item.name for item in reparsed.tensors],
        )
        self.assertEqual(v1.quantization_bits, reparsed.quantization_bits)
        self.assertEqual(
            [item.scale for item in v1.tensors],
            [item.scale for item in reparsed.tensors],
        )
        self.assertTrue(
            all(row["scale_truncation_residual_milli"] == 0 for row in sidecar["tensors"])
        )

    def test_v2_downgrade_records_all_losses_in_stable_order(self) -> None:
        v2 = parse_v2(self.v2_document, self.names)
        _, _, sidecar = downgrade_v2_to_v1(v2)
        rows = sidecar["tensors"]
        self.assertEqual([row["name"] for row in rows], sorted(row["name"] for row in rows))
        asymmetric = next(row for row in rows if row["name"] == "mlp.up.weight")
        self.assertEqual(asymmetric["named_rounding_mode"], "nearest_away")
        self.assertNotEqual(asymmetric["asymmetric_zero_point_offset"], 0)
        self.assertIsNotNone(asymmetric["discarded_quantile_range"])
        self.assertIn("scale_truncation_residual_milli", asymmetric)
        self.assertIn("asymmetric_scheme_approximated_as_symmetric", asymmetric["loss_codes"])
        self.assertIn("named_rounding_mode_discarded", sidecar["losses"])

    def test_parser_error_codes(self) -> None:
        duplicate = copy.deepcopy(self.v2_document)
        duplicate["tensors"].append(copy.deepcopy(duplicate["tensors"][0]))
        with self.assertRaisesRegex(QuantCalError, "duplicate") as duplicate_error:
            parse_v2(duplicate, self.names)
        self.assertEqual(duplicate_error.exception.code, "E_FORMAT_INVALID")

        bad_zero = copy.deepcopy(self.v2_document)
        bad_zero["tensors"][0]["zero_point"] = 999
        with self.assertRaises(QuantCalError) as zero_error:
            parse_v2(bad_zero, self.names)
        self.assertEqual(zero_error.exception.code, "E_ZERO_POINT_INVALID")

        bad_range = copy.deepcopy(self.v2_document)
        bad_range["tensors"][0]["quantile_range"] = [2.0, 1.0]
        with self.assertRaises(QuantCalError) as range_error:
            parse_v2(bad_range, self.names)
        self.assertEqual(range_error.exception.code, "E_QUANT_RANGE_INVALID")

        lines = self.v1_text.splitlines()
        broken = "\n".join(lines[:2] + [lines[2][:-1]]) + "\n"
        with self.assertRaises(QuantCalError) as format_error:
            parse_v1(broken, self.names)
        self.assertEqual(format_error.exception.code, "E_FORMAT_INVALID")

    def test_v1_render_rejects_asymmetric(self) -> None:
        v2 = parse_v2(self.v2_document, self.names)
        with self.assertRaises(QuantCalError):
            render_v1(v2)

    def test_v1_accepts_positive_saturation_cap_below_qmax(self) -> None:
        lines = self.v1_text.splitlines()
        width = 8
        lines[1] = lines[1][:-width] + str(100).rjust(width)
        parsed = parse_v1("\n".join(lines) + "\n", self.names)
        self.assertEqual(parsed.tensors[0].saturation_max, 100)
        upgraded, _ = upgrade_v1_to_v2(parsed)
        _, text, _ = downgrade_v2_to_v1(upgraded)
        reparsed = parse_v1(text, self.names)
        self.assertEqual(reparsed.tensors[0].saturation_max, 100)


if __name__ == "__main__":
    unittest.main()
