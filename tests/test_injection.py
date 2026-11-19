from __future__ import annotations

import unittest
from dataclasses import replace

from spectral_diag.config import RunConfig
from spectral_diag.pipeline import execute_pipeline


class InjectionTests(unittest.TestCase):
    def _run(self, injection: dict) -> dict:
        return execute_pipeline(replace(RunConfig(), injections=(injection,)))

    def test_collinear_samples_are_rejected(self) -> None:
        result = self._run({"kind": "collinear_sample", "params": {}})
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error"]["code"], "E_MATRIX_SINGULAR")
        self.assertIsNone(result["matrix"]["matrix"])
        self.assertEqual(result["first_affected"]["sample_index"], 0)
        self.assertEqual(result["first_affected"]["component"], 0)

    def test_zero_white_point_is_located(self) -> None:
        result = self._run({"kind": "zero_white_point", "params": {"component": 1}})
        self.assertEqual(result["error"]["code"], "E_WHITE_POINT")
        self.assertEqual(result["first_affected"]["component"], 1)
        self.assertEqual(result["first_affected"]["component_name"], "Y")

    def test_quant_range_is_located(self) -> None:
        result = self._run(
            {"kind": "quant_out_of_range", "params": {"sample_index": 7, "component": 1}}
        )
        self.assertEqual(result["error"]["code"], "E_QUANT_RANGE")
        self.assertEqual(result["first_affected"]["sample_index"], 7)
        self.assertEqual(result["first_affected"]["component"], 1)

    def test_invalid_observer_is_located(self) -> None:
        result = self._run({"kind": "observer_invalid", "params": {"observer": "invalid"}})
        self.assertEqual(result["error"]["code"], "E_OBSERVER_INVALID")
        self.assertEqual(result["first_affected"]["observation_kind"], "observer_marker")

    def test_low_integration_density_is_rejected(self) -> None:
        result = self._run({"kind": "low_integration_density", "params": {"density_nm": 20.0}})
        self.assertEqual(result["error"]["code"], "E_INTEGRATION_DENSITY")
        self.assertIn("maximum_spacing_nm", result["error"]["details"])

    def test_injection_config_is_recorded(self) -> None:
        injection = {"kind": "zero_white_point", "params": {"component": 2}}
        result = self._run(injection)
        self.assertEqual(result["injections"], [injection])
        self.assertIsNotNone(result["injection_sha256"])


if __name__ == "__main__":
    unittest.main()
