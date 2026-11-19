from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

import spectral_workbench.pipeline as pipeline_module
from spectral_workbench.config import (
    CalibrationConfig,
    ConcentrationBiasInjection,
    InjectionConfig,
    PipelineConfig,
    UnmixConfig,
)
from spectral_workbench.errors import ErrorCode, WorkbenchError
from spectral_workbench.generator import generate_cube, load_endmembers, load_manifest
from spectral_workbench.pipeline import PipelineRunner
from tests.helpers import small_cube_config, small_pipeline_config


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.cube = Path(cls.temporary.name) / "cube"
        generate_cube(small_cube_config(), InjectionConfig(), cls.cube)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_full_pipeline_closure_and_compensation_improvement(self) -> None:
        output = Path(self.temporary.name) / "run_full"
        result = PipelineRunner(self.cube, small_pipeline_config()).run(output, verify_closure=True)
        self.assertTrue(result.report["closure"]["verified"])
        self.assertTrue(result.report["metrics"]["compensation_evaluation"]["improved"])
        self.assertLess(
            result.report["metrics"]["post_compensation"]["overall_absolute_error_mean"],
            result.report["metrics"]["pre_compensation"]["overall_absolute_error_mean"],
        )
        self.assertLess(result.report["drift"]["max_estimate_residual_px"], 0.05)
        self.assertEqual(result.post_unmix.concentrations.shape[0], 4)

    def test_identical_rerun_reuses_all_stages(self) -> None:
        output = Path(self.temporary.name) / "run_reuse"
        config = small_pipeline_config()
        PipelineRunner(self.cube, config).run(output)
        second = PipelineRunner(self.cube, config).run(output)
        reuse = second.report["incremental"]["stage_reuse"]
        self.assertEqual(reuse["calibration"]["computed_frames"], 0)
        self.assertEqual(reuse["drift"]["computed_frames"], 0)
        self.assertEqual(reuse["unmix"]["computed_frames"], 0)
        self.assertEqual(reuse["metrics"]["computed_frames"], 0)

    def test_endmember_change_reuses_calibration_and_drift(self) -> None:
        output = Path(self.temporary.name) / "run_incremental"
        PipelineRunner(self.cube, small_pipeline_config()).run(output)
        manifest = load_manifest(self.cube, verify_hashes=False)
        endmembers = load_endmembers(self.cube, manifest)
        changed = endmembers.copy()
        changed[:, 0] *= 1.01
        override = Path(self.temporary.name) / "override.npy"
        np.save(override, changed)
        config = PipelineConfig(
            unmix=UnmixConfig(cond_warn=1.0e9, endmember_file=str(override)),
            drift=small_pipeline_config().drift,
            concentration_tolerance=0.08,
        )
        result = PipelineRunner(self.cube, config).run(output, verify_closure=True)
        reuse = result.report["incremental"]["stage_reuse"]
        self.assertEqual(reuse["calibration"]["computed_frames"], 0)
        self.assertEqual(reuse["drift"]["computed_frames"], 0)
        self.assertEqual(reuse["unmix"]["reused_frames"], 0)
        self.assertTrue(result.report["closure"]["verified"])

    def test_compensation_toggle_reuses_upstream_stages(self) -> None:
        output = Path(self.temporary.name) / "run_toggle"
        PipelineRunner(self.cube, small_pipeline_config(compensate=True)).run(output)
        result = PipelineRunner(self.cube, small_pipeline_config(compensate=False)).run(output, verify_closure=True)
        reuse = result.report["incremental"]["stage_reuse"]
        self.assertEqual(reuse["calibration"]["computed_frames"], 0)
        self.assertEqual(reuse["drift"]["computed_frames"], 0)
        self.assertEqual(reuse["unmix"]["computed_frames"], 0)
        self.assertTrue(result.report["closure"]["verified"])

    def test_calibration_parameter_change_reuses_content_identical_downstream(self) -> None:
        output = Path(self.temporary.name) / "run_calibration_param"
        baseline = small_pipeline_config()
        PipelineRunner(self.cube, baseline).run(output)
        changed = PipelineConfig(
            calibration=CalibrationConfig(flat_epsilon_dn=2.0e-6),
            unmix=baseline.unmix,
            drift=baseline.drift,
            concentration_tolerance=baseline.concentration_tolerance,
        )
        result = PipelineRunner(self.cube, changed).run(output, verify_closure=True)
        reuse = result.report["incremental"]["stage_reuse"]
        self.assertGreater(reuse["calibration"]["computed_frames"], 0)
        self.assertEqual(reuse["unmix"]["computed_frames"], 0)
        self.assertEqual(reuse["metrics"]["computed_frames"], 0)
        self.assertTrue(result.report["closure"]["verified"])

    def test_unmix_order_change_recomputes_and_closes(self) -> None:
        output = Path(self.temporary.name) / "run_order"
        baseline = small_pipeline_config()
        PipelineRunner(self.cube, baseline).run(output)
        changed = PipelineConfig(
            unmix=UnmixConfig(order=3, cond_warn=1.0e9),
            drift=baseline.drift,
            concentration_tolerance=baseline.concentration_tolerance,
        )
        result = PipelineRunner(self.cube, changed).run(output, verify_closure=True)
        reuse = result.report["incremental"]["stage_reuse"]
        self.assertEqual(reuse["calibration"]["computed_frames"], 0)
        self.assertEqual(reuse["drift"]["computed_frames"], 0)
        self.assertEqual(reuse["unmix"]["reused_frames"], 0)
        self.assertEqual(result.post_unmix.concentrations.shape[0], 3)
        self.assertTrue(result.report["closure"]["verified"])

    def test_closure_mismatch_locates_first_pixel(self) -> None:
        output = Path(self.temporary.name) / "run_closure"
        config = small_pipeline_config()
        result = PipelineRunner(self.cube, config).run(output)
        real_unmix = pipeline_module.unmix_frame

        def altered_unmix(*args, **kwargs):
            value = real_unmix(*args, **kwargs)
            concentrations = value.concentrations.copy()
            concentrations[0, 0, 0] += 0.01
            return replace(value, concentrations=concentrations)

        with mock.patch("spectral_workbench.pipeline.unmix_frame", side_effect=altered_unmix):
            with self.assertRaises(WorkbenchError) as context:
                PipelineRunner(self.cube, config).verify_closure(result)
        self.assertEqual(context.exception.code, ErrorCode.E_CLOSURE_MISMATCH)
        self.assertEqual(context.exception.details["divergence"]["pixel"], [0, 0])

    def test_known_bias_exposes_first_violation_and_components(self) -> None:
        cube = Path(self.temporary.name) / "bias_cube"
        output = Path(self.temporary.name) / "bias_run"
        injection = InjectionConfig(
            concentration_bias=ConcentrationBiasInjection(
                fluorophore=0,
                delta=1.0,
                frame=0,
                y0=5,
                x0=5,
                height=4,
                width=4,
            )
        )
        generate_cube(small_cube_config(seed=29), injection, cube)
        result = PipelineRunner(cube, small_pipeline_config(tolerance=0.08)).run(output)
        violation = result.report["metrics"]["post_compensation"]["first_tolerance_violation"]
        self.assertIsNotNone(violation)
        self.assertEqual(violation["frame"], 0)
        self.assertEqual(violation["pixel"], [5, 5])
        self.assertEqual(len(violation["fluorophores"]), 4)
        self.assertGreater(violation["absolute_errors"][0], 0.08)
        self.assertEqual(len(violation["fluorophores"][0]["components"]), 4)

    def test_no_drift_reports_compensation_not_improved(self) -> None:
        cube = Path(self.temporary.name) / "flat_cube"
        output = Path(self.temporary.name) / "flat_run"
        generate_cube(small_cube_config(seed=31, drift=0.0), InjectionConfig(), cube)
        result = PipelineRunner(cube, small_pipeline_config(compensate=True)).run(output)
        evaluation = result.report["metrics"]["compensation_evaluation"]
        self.assertFalse(evaluation["improved"])
        self.assertTrue(any(item["code"] == "COMPENSATION_NOT_IMPROVED" for item in result.report["warnings"]))


if __name__ == "__main__":
    unittest.main()
