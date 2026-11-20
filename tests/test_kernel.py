from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from quantcal.calibration import (
    calibrate_quantiles,
    derive_scale_zero,
    exchange_scaled_result,
    parse_samples,
    parse_tensor_specs,
    tensor_values_and_refs,
)
from quantcal.config import parse_config
from quantcal.quantize import quantize_tensor
from quantcal.util import load_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest = load_json(SAMPLE / "tensors.json")
        config_document = load_json(SAMPLE / "config.json")
        rows = [
            json.loads(line)
            for line in (SAMPLE / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.specs = parse_tensor_specs(manifest)
        cls.samples = parse_samples(rows, cls.specs)
        cls.config = parse_config(config_document)

    def _calibrate(self, name: str, resolved=None, version: str = "v2"):
        resolved = resolved or self.config.resolve(name)
        values, refs, layout = tensor_values_and_refs(name, self.samples)
        quantiles = calibrate_quantiles(name, values, refs, layout, resolved)
        scale = exchange_scaled_result(derive_scale_zero(name, quantiles, resolved), version)
        return resolved, values, refs, layout, quantiles, scale

    def test_quantiles_are_deterministic_and_metadata_is_fixed(self) -> None:
        resolved, _, refs, layout, first, _ = self._calibrate("attn.q.weight")
        _, _, _, _, second, _ = self._calibrate("attn.q.weight")
        self.assertEqual(first["quantile_lower"], second["quantile_lower"])
        self.assertEqual(first["quantile_upper"], second["quantile_upper"])
        self.assertEqual(first["sample_ids"], sorted(refs))
        self.assertEqual(first["quantile_metadata"]["interpolation"], "linear_type7")
        self.assertIn("duplicate_multiplicity", first["quantile_metadata"]["ties"])
        self.assertEqual([item["flat_index"] for item in layout], list(range(len(layout))))
        self.assertEqual(resolved.quantile_lower_permille, 10)

    def test_zero_range_guard_is_recorded(self) -> None:
        _, values, _, _, _, scale = self._calibrate("zero.weight")
        self.assertTrue(all(value == 0.0 for value in values))
        self.assertEqual(scale["scale"], 0.001)
        self.assertIn("zero_abs_range", scale["scale_guard"])

    def test_bits_fractional_width_and_rounding_change_kernel_outputs(self) -> None:
        base = self.config.resolve("attn.q.weight")
        results = {}
        for bits in (8, 12):
            resolved = replace(base, quantization_bits=bits)
            _, values, _, layout, _, scale = self._calibrate("attn.q.weight", resolved)
            results[f"bits{bits}"] = quantize_tensor(
                "attn.q.weight", values, layout, resolved, scale, "v2"
            )
        self.assertNotEqual(
            results["bits8"]["effective_scale"], results["bits12"]["effective_scale"]
        )
        self.assertNotEqual(
            results["bits8"]["quantized_values"], results["bits12"]["quantized_values"]
        )
        self.assertNotEqual(
            results["bits8"]["dequantized_values"], results["bits12"]["dequantized_values"]
        )

        fractional = {}
        for fraction in (4, 16):
            resolved = replace(base, fixed_fractional_bits=fraction)
            _, values, _, layout, _, scale = self._calibrate("attn.q.weight", resolved)
            fractional[fraction] = quantize_tensor(
                "attn.q.weight", values, layout, resolved, scale, "v2"
            )
        self.assertNotEqual(
            fractional[4]["quantized_values"], fractional[16]["quantized_values"]
        )
        self.assertNotEqual(
            fractional[4]["dequantized_values"], fractional[16]["dequantized_values"]
        )

        rounding = {}
        for mode in ("nearest_even", "nearest_away", "floor", "truncate"):
            resolved = replace(base, rounding_mode=mode)
            _, values, _, layout, _, scale = self._calibrate("attn.q.weight", resolved)
            rounding[mode] = quantize_tensor(
                "attn.q.weight", values, layout, resolved, scale, "v2"
            )
        self.assertNotEqual(
            rounding["nearest_even"]["quantized_values"],
            rounding["floor"]["quantized_values"],
        )
        self.assertNotEqual(
            rounding["nearest_even"]["dequantized_values"],
            rounding["floor"]["dequantized_values"],
        )
        self.assertEqual(rounding["floor"]["rounding_mode"], "floor")

    def test_saturation_positions_and_strategy(self) -> None:
        resolved = self.config.resolve("mlp.up.weight")
        _, values, _, layout, _, scale = self._calibrate("mlp.up.weight", resolved)
        standard = quantize_tensor("mlp.up.weight", values, layout, resolved, scale, "v2")
        self.assertGreater(standard["clipped_count"], 0)
        positions = [item["flat_index"] for item in standard["clipped_positions"]]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(
            all(item["direction"] in {"lower", "upper"} for item in standard["clipped_positions"])
        )

        symmetric = replace(resolved, saturation_strategy="saturate_symmetric")
        narrowed = quantize_tensor("mlp.up.weight", values, layout, symmetric, scale, "v2")
        self.assertEqual(narrowed["clamp_bounds"], [-127, 127])
        self.assertNotEqual(narrowed["clipped_count"], standard["clipped_count"])
        self.assertGreater(narrowed["clipped_count"], standard["clipped_count"])


if __name__ == "__main__":
    unittest.main()
