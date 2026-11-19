from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectral_workbench.config import InjectionConfig, UnmixConfig
from spectral_workbench.errors import ErrorCode, WorkbenchError
from spectral_workbench.generator import generate_cube, load_endmembers, load_manifest
from spectral_workbench.unmix import build_endmember_model, concentration_error_components, unmix_frame
from tests.helpers import small_cube_config


class UnmixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "cube"
        generate_cube(small_cube_config(), InjectionConfig(), cls.root)
        cls.manifest = load_manifest(cls.root, verify_hashes=False)
        cls.endmembers = load_endmembers(cls.root, cls.manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_nnls_recovers_nonnegative_synthetic_abundance(self) -> None:
        config = UnmixConfig(cond_warn=1.0e9)
        model = build_endmember_model(self.endmembers, self.manifest["fluorophore_names"], config)
        expected = np.array(
            [
                [0.2, 1.5, 0.0, 0.6, 0.0, 1.1, 0.9, 0.0, 0.35, 1.3],
                [1.1, 0.0, 0.4, 0.2, 0.8, 0.0, 0.1, 1.4, 0.0, 0.5],
                [0.0, 0.3, 1.2, 0.9, 0.1, 0.5, 0.0, 0.2, 1.1, 0.0],
                [0.6, 0.1, 0.0, 0.0, 1.4, 0.7, 0.4, 0.0, 0.5, 0.9],
            ],
            dtype=np.float64,
        )
        cube = (model.endmembers @ expected).reshape(model.endmembers.shape[0], 2, 5)
        result = unmix_frame(
            cube,
            np.zeros((2, 5), dtype=np.uint8),
            model,
            config,
            quantization_upper_bound=1.0,
        )
        recovered = result.concentrations.reshape(4, 10)
        np.testing.assert_allclose(recovered, expected, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(result.residual, 0.0, atol=1.0e-3)

    def test_condition_threshold_marks_pixels(self) -> None:
        model = build_endmember_model(
            self.endmembers,
            self.manifest["fluorophore_names"],
            UnmixConfig(cond_warn=0.01),
        )
        cube = np.ones((8, 3, 3), dtype=np.float32)
        result = unmix_frame(
            cube,
            np.zeros((3, 3), dtype=np.uint8),
            model,
            UnmixConfig(cond_warn=0.01),
            quantization_upper_bound=1.0,
        )
        self.assertGreater(result.ill_conditioned_pixels, 0)
        components = concentration_error_components(model, result, frame_index=0, y=1, x=1)
        for item in components["fluorophores"]:
            self.assertGreaterEqual(item["error_bound"], 0.0)
            self.assertLessEqual(item["lower"], item["estimate"])
            self.assertGreaterEqual(item["upper"], item["estimate"])
            self.assertEqual(
                set(item["components"]),
                {
                    "calibration_quantization",
                    "unmix_residual",
                    "condition_amplification",
                    "float_roundoff",
                },
            )

    def test_degenerate_endmembers_raise_unmix_condition_error(self) -> None:
        matrix = self.endmembers.copy()
        matrix[:, 1] = matrix[:, 0]
        with self.assertRaises(WorkbenchError) as context:
            build_endmember_model(matrix, self.manifest["fluorophore_names"], UnmixConfig())
        self.assertEqual(context.exception.code, ErrorCode.E_UNMIX_COND)


if __name__ == "__main__":
    unittest.main()
