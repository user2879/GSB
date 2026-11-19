from __future__ import annotations

import unittest

from spectral_workspace.errors import DomainError
from spectral_workspace.injection import (
    injection_hash,
    run_parameter_injection_case,
    validate_injection_config,
)
from spectral_workspace.synth import generate_sample_scene


class InjectionCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = generate_sample_scene()

    def test_parameter_level_error_codes_and_pointers(self) -> None:
        cases = [
            (
                {
                    "schema": "spectral-injection/v1",
                    "reflectance_out_of_range": {
                        "material_id": "mat_oxide",
                        "sample_index": 17,
                        "value": 1.25,
                    },
                },
                "E_SPECTRUM_RANGE",
                "/materials/mat_oxide/reflectance/17",
            ),
            (
                {
                    "schema": "spectral-injection/v1",
                    "white_point_zero": {"value": [0.0, 1.0, 1.0]},
                },
                "E_WHITE_POINT",
                "/scene/camera_white_point_xyz/0",
            ),
            (
                {
                    "schema": "spectral-injection/v1",
                    "observer_invalid": {"value": "INVALID"},
                },
                "E_OBSERVER_INVALID",
                "/scene/observer",
            ),
        ]
        for config, code, pointer in cases:
            outcome = run_parameter_injection_case(self.sample, config)
            self.assertFalse(outcome["ok"])
            self.assertEqual(outcome["error"]["code"], code)
            self.assertEqual(outcome["error"]["pointer"], pointer)

    def test_injection_config_is_serialisable_and_stable(self) -> None:
        config = {
            "schema": "spectral-injection/v1",
            "name": "stable",
            "stage_failure": {
                "operation_version": 4,
                "stage_kind": "material_color",
                "material_id": "mat_oxide",
                "delta_xyz": [0.25, 0.0, 0.0],
            },
        }
        cleaned = validate_injection_config(config)
        self.assertEqual(cleaned["stage_failure"]["light_id"], None)
        self.assertEqual(injection_hash(config), injection_hash(cleaned))

    def test_invalid_injection_config(self) -> None:
        with self.assertRaises(DomainError) as context:
            validate_injection_config(
                {"schema": "spectral-injection/v1", "unknown_key": {}}
            )
        self.assertEqual(context.exception.code, "E_INJECT_CONFIG")


if __name__ == "__main__":
    unittest.main()
