from __future__ import annotations

import unittest

from quantcalib.calibration import calibrate_group
from quantcalib.distillation import analyze_group, build_group_distillation, distill_vectors
from quantcalib.model import CalibrationSample, CalibrationSet, DistillationConfig
from quantcalib.quantization import quantize_group

from tests.test_calibration_quantization import make_cfg, make_layer


class DistillationTests(unittest.TestCase):
    def _analysis(self, weights: list[float]):
        layer = make_layer(weights)
        calibration = CalibrationSet(
            input_dim=2,
            output_dim=2,
            samples=(
                CalibrationSample("s0", (0.5, -0.25), (0.0, 0.0), 0),
                CalibrationSample("s1", (-1.0, 0.75), (0.1, -0.2), 1),
            ),
        )
        cfg = make_cfg()
        distill_cfg = DistillationConfig(2.0, 4, 0.6, 0.02)
        group = calibrate_group(layer, cfg, 0, calibration_sample_ids=calibration.sample_ids)
        quant = quantize_group(
            layer,
            cfg,
            group,
            batch_point_budget=2,
            calibration_sample_ids=list(calibration.sample_ids),
        )
        analysis = analyze_group(
            layer=layer,
            cfg=cfg,
            calibration=calibration,
            calibration_group=group,
            quantization_group=quant,
            distill_cfg=distill_cfg,
        )
        distillation = build_group_distillation(
            layer=layer,
            calibration=calibration,
            error_analysis=analysis,
            distill_cfg=distill_cfg,
        )
        return analysis, distillation

    def test_zero_truth_uses_absolute_branch(self) -> None:
        analysis, _ = self._analysis([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(record["before"]["truth_zero"] for record in analysis["records"]))
        self.assertTrue(
            all(record["before"]["decision_branch"] == "absolute_zero_truth" for record in analysis["records"])
        )

    def test_distillation_moves_toward_raw_output(self) -> None:
        analysis, distillation = self._analysis([2.5, -0.98, -0.09, -0.51])
        self.assertTrue(any(record["before"]["exceeded"] for record in analysis["records"]))
        self.assertEqual(analysis["attribution_priority"][0], "saturation_truncation")
        for sample in distillation["samples"]:
            before = sum(
                abs(q - r)
                for q, r in zip(sample["quantized_output"], sample["raw_output"])
            )
            after = sum(
                abs(c - r)
                for c, r in zip(sample["corrected_output"], sample["raw_output"])
            )
            self.assertLessEqual(after, before)

    def test_temperature_and_steps_change_correction(self) -> None:
        raw = [1.0, -0.5, 0.25]
        quant = [0.8, -0.4, 0.1]
        cold, _ = distill_vectors(raw, quant, DistillationConfig(0.5, 3, 0.5, 0.01))
        hot, _ = distill_vectors(raw, quant, DistillationConfig(4.0, 3, 0.5, 0.01))
        more_steps, _ = distill_vectors(raw, quant, DistillationConfig(0.5, 8, 0.5, 0.01))
        self.assertNotEqual(cold, hot)
        self.assertNotEqual(cold, more_steps)


if __name__ == "__main__":
    unittest.main()
