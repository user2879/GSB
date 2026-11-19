from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spectral_workbench.config import (
    CenterShiftInjection,
    DarkInconsistencyInjection,
    DegenerateEndmemberInjection,
    InjectionConfig,
    SaturationOverloadInjection,
)
from spectral_workbench.errors import ErrorCode, WorkbenchError
from spectral_workbench.generator import generate_cube
from spectral_workbench.pipeline import PipelineRunner
from tests.helpers import small_cube_config, small_pipeline_config


class InjectedErrorChainTests(unittest.TestCase):
    def _run_with_injection(self, name: str, injection: InjectionConfig) -> WorkbenchError:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        generate_cube(small_cube_config(seed=41), injection, root / "cube")
        with self.assertRaises(WorkbenchError) as context:
            PipelineRunner(root / "cube", small_pipeline_config()).run(root / "run")
        return context.exception

    def test_dark_inconsistency(self) -> None:
        error = self._run_with_injection(
            "dark",
            InjectionConfig(
                dark_inconsistency=DarkInconsistencyInjection(offset_dn=2000.0, band_start=0, band_end=8)
            ),
        )
        self.assertEqual(error.code, ErrorCode.E_DARK_GAIN)

    def test_degenerate_endmembers(self) -> None:
        error = self._run_with_injection(
            "degenerate",
            InjectionConfig(degenerate_endmembers=DegenerateEndmemberInjection(source=0, target=1)),
        )
        self.assertEqual(error.code, ErrorCode.E_UNMIX_COND)

    def test_center_shift_bias(self) -> None:
        error = self._run_with_injection(
            "shift",
            InjectionConfig(center_shift=CenterShiftInjection(dx_px=3.0, dy_px=2.0, frame_start=0, frame_end=8)),
        )
        self.assertEqual(error.code, ErrorCode.E_DRIFT_BIAS)

    def test_saturation_overload(self) -> None:
        error = self._run_with_injection(
            "saturation",
            InjectionConfig(
                saturation_overload=SaturationOverloadInjection(frame=0, y0=0, x0=0, height=16, width=16)
            ),
        )
        self.assertEqual(error.code, ErrorCode.E_CUBE_FORMAT)


if __name__ == "__main__":
    unittest.main()
