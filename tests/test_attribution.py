from __future__ import annotations

import unittest

from spectral_diag.attribution import verify_attribution
from spectral_diag.config import RunConfig
from spectral_diag.errors import DiagnosticError, ErrorCode
from spectral_diag.pipeline import execute_pipeline


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = RunConfig()
        cls.result = execute_pipeline(cls.config)

    def test_ledger_reconciles_and_verifies(self) -> None:
        attribution = self.result["attribution"]
        verify_attribution(attribution)
        self.assertTrue(attribution["additivity"]["reconciled"])
        factor_sum = sum(
            attribution["contributions"][factor]["shapley_delta"]
            for factor in ("quantization", "distribution", "integration")
        )
        self.assertAlmostEqual(factor_sum, attribution["additivity"]["total_delta"], places=10)
        for factor in ("quantization", "distribution", "integration"):
            self.assertIn("standalone_residual", attribution["contributions"][factor])
            self.assertIn("interaction_allocation", attribution["contributions"][factor])

    def test_quantization_matrix_offset_is_reported(self) -> None:
        quantization = self.result["attribution"]["perturbations"]["quantization"]
        self.assertTrue(quantization["matrix_offset"]["available"])
        self.assertEqual(len(quantization["matrix_offset"]["element_offsets"]), 3)
        self.assertIn("condition_delta", quantization["matrix_offset"])

    def test_tampered_ledger_raises_attribution_mismatch(self) -> None:
        attribution = dict(self.result["attribution"])
        attribution["contributions"] = dict(attribution["contributions"])
        attribution["contributions"]["quantization"] = dict(attribution["contributions"]["quantization"])
        attribution["contributions"]["quantization"]["shapley_delta"] = 99.0
        with self.assertRaises(DiagnosticError) as context:
            verify_attribution(attribution)
        self.assertEqual(context.exception.code, ErrorCode.ATTRIB_MISMATCH)


if __name__ == "__main__":
    unittest.main()
