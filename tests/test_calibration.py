from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectral_workbench.calibration import calibrate_frame
from spectral_workbench.config import CalibrationConfig, InjectionConfig
from spectral_workbench.errors import ErrorCode, WorkbenchError
from spectral_workbench.generator import generate_cube, load_frame_array, load_manifest
from tests.helpers import small_cube_config


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "cube"
        generate_cube(small_cube_config(), InjectionConfig(), cls.root)
        cls.manifest = load_manifest(cls.root, verify_hashes=False)
        cls.height, cls.width = cls.manifest["shape"][1:]
        cls.record = cls.manifest["frames"][0]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _inputs(self):
        raw = load_frame_array(self.root, self.record, "raw", (self.height, self.width))
        dark = load_frame_array(self.root, self.record, "dark", (self.height, self.width))
        flat = load_frame_array(self.root, self.record, "flat", (self.height, self.width))
        return raw, dark, flat

    def test_calibration_is_deterministic_and_audited(self) -> None:
        raw, dark, flat = self._inputs()
        first = calibrate_frame(raw, dark, flat, CalibrationConfig())
        second = calibrate_frame(raw, dark, flat, CalibrationConfig())
        self.assertEqual(first.corrected.dtype, np.float32)
        self.assertEqual(first.corrected.tobytes(), second.corrected.tobytes())
        self.assertEqual(first.record_bytes(), second.record_bytes())
        self.assertGreater(first.quantization_upper_bound, 0.0)
        self.assertEqual(first.invalid_pixels, 0)

    def test_zero_flat_and_saturation_are_counted_not_silent(self) -> None:
        raw, dark, flat = self._inputs()
        flat = flat.copy()
        raw = raw.copy()
        flat[0, 0] = dark[0, 0]
        raw[1, 1] = 65535
        result = calibrate_frame(raw, dark, flat, CalibrationConfig())
        self.assertEqual(result.zero_flat_pixels, 1)
        self.assertEqual(result.saturated_pixels, 1)
        self.assertEqual(result.invalid_pixels, 2)
        self.assertEqual(int(result.invalid_mask[0, 0]), 1)
        self.assertEqual(int(result.invalid_mask[1, 1]), 1)
        self.assertEqual(float(result.corrected[0, 0]), 0.0)
        self.assertEqual(result.record["invalid_pixels"], 2)

    def test_inconsistent_dark_raises_business_error(self) -> None:
        raw, dark, flat = self._inputs()
        dark = np.clip(dark.astype(np.int32) + 2000, 0, 65535).astype(np.uint16)
        with self.assertRaises(WorkbenchError) as context:
            calibrate_frame(raw, dark, flat, CalibrationConfig())
        self.assertEqual(context.exception.code, ErrorCode.E_DARK_GAIN)


if __name__ == "__main__":
    unittest.main()
