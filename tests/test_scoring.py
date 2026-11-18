"""定点量化、双档位分数分解与边界决策测试。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proofread_app.app import ProofreadApp
from proofread_app.config import AppConfig
from proofread_app.detection import segment_sample, detect_candidates
from proofread_app.scoring import Quantizer, score_all, DECISION_APPLY, DECISION_SKIP


class QuantizerTests(unittest.TestCase):
    def test_quantizer_uses_fixed_point_scale(self) -> None:
        quant = Quantizer(levels=32, frac_bits=8, precision="float64")
        self.assertEqual(quant.scale, 256.0)
        result = quant.quantize(1.0)
        self.assertEqual(result.quantized_int, 256)
        self.assertEqual(result.dequantized, 1.0)

    def test_quantizer_clamps_to_levels(self) -> None:
        quant = Quantizer(levels=32, frac_bits=8, precision="float64")
        self.assertEqual(quant.quantize(5.0).quantized_int, 256)
        self.assertEqual(quant.quantize(-1.0).quantized_int, 0)

    def test_quantizer_is_deterministic_across_precision(self) -> None:
        q32 = Quantizer(levels=16, frac_bits=8, precision="float32")
        q64 = Quantizer(levels=16, frac_bits=8, precision="float64")
        self.assertEqual(q32.quantize(0.37).quantized_int,
                         q64.quantize(0.37).quantized_int)


class ScoreBreakdownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        import shutil
        for name in ("samples", "resources", "config"):
            shutil.copytree(ROOT / name, self.tmp / name)
        self.app = ProofreadApp(str(self.tmp))

    def test_breakdown_contains_fixed_point_intermediates(self) -> None:
        sample = next(s for s in self.app.samples if s.sample_id == "S007")
        candidates, _ = detect_candidates(
            sample, segment_sample(sample, self.app.config),
            self.app.resource, self.app.config)
        breakdown = score_all(candidates, self.app.resource, self.app.scoring,
                              self.app.config, "float32")[0]
        for field in ("detect_quant", "confidence_quant"):
            quant = getattr(breakdown, field)
            self.assertIn("raw", quant)
            self.assertIn("quantized_int", quant)
            self.assertIn("scale", quant)
            self.assertIn("dequantized", quant)
        self.assertEqual(breakdown.detect_quant["scale"],
                         float(1 << self.app.config.fixed_point_bits))

    def test_two_precision_brackets_both_produce_scores(self) -> None:
        sample = next(s for s in self.app.samples if s.sample_id == "S007")
        candidates, _ = detect_candidates(
            sample, segment_sample(sample, self.app.config),
            self.app.resource, self.app.config)
        fp32 = score_all(candidates, self.app.resource, self.app.scoring,
                         self.app.config, "float32")
        fp64 = score_all(candidates, self.app.resource, self.app.scoring,
                         self.app.config, "float64")
        self.assertEqual(len(fp32), len(fp64))
        self.assertEqual({b.precision for b in fp32}, {"float32"})
        self.assertEqual({b.precision for b in fp64}, {"float64"})

    def test_boundary_equality_uses_type_priority(self) -> None:
        # 构造总分恰好等于阈值的场景，验证边界标注与固定优先级
        sample = next(s for s in self.app.samples if s.sample_id == "S006")
        candidates, _ = detect_candidates(
            sample, segment_sample(sample, self.app.config),
            self.app.resource, self.app.config)
        fp64 = score_all(candidates, self.app.resource, self.app.scoring,
                         self.app.config, "float64")
        boundary = [b for b in fp64 if b.boundary]
        self.assertTrue(boundary)
        for item in boundary:
            self.assertEqual(item.decision, DECISION_APPLY)


class ConfigValidationTests(unittest.TestCase):
    def test_non_positive_ngram_rejected(self) -> None:
        from proofread_app.errors import ProofreadError
        from proofread_app.config import build_config
        base = {
            "ngram_order": 0, "similarity_threshold": 0.2, "fixed_point_bits": 8,
            "quant_levels": 32, "aggregation_weights": {"detect": 0.5, "confidence": 0.5},
            "precision": "float32", "tolerance": 1e-6, "max_workers": 2,
            "snapshot_interval": 5, "decision_threshold": 0.3,
        }
        with self.assertRaises(ProofreadError) as ctx:
            build_config(base)
        self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
