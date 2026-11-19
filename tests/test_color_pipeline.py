from __future__ import annotations

import copy
import unittest

from spectral_workspace.color import (
    adapt_xyz,
    adaptation_matrix,
    spectral_to_xyz,
    xyz_to_target,
)
from spectral_workspace.errors import DomainError
from spectral_workspace.model import parse_scene
from spectral_workspace.pipeline import (
    adapted_stage_id,
    build_pipeline,
    spectral_stage_id,
)
from spectral_workspace.standards import mat3_vec
from spectral_workspace.synth import generate_sample_scene


class ColorPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = parse_scene(generate_sample_scene())
        self.material = self.scene.materials["mat_oxide"]
        self.light = self.scene.lights["light_day"]

    def test_spectral_integrals_are_separated_and_normalised(self) -> None:
        product = spectral_to_xyz(self.material, self.light, "CIE1931_2")
        self.assertEqual(product["integration_step_nm"], 5)
        self.assertEqual(product["normalization"]["method"], "normalize_light_Y_to_100")
        self.assertAlmostEqual(product["normalization"]["light_Y_integral"], 100.0 / product["normalization"]["k"], places=10)
        self.assertAlmostEqual(product["source_white_raw_xyz"][1], 100.0, places=10)
        self.assertEqual(len(product["terms"]["light_integral_xyz"]), 3)
        self.assertEqual(len(product["terms"]["reflectance_x_light_integral_xyz"]), 3)
        self.assertNotEqual(
            [round(value, 10) for value in product["xyz"]], [0.0, 0.0, 0.0]
        )

    def test_two_observers_produce_distinguishable_xyz_and_hashes(self) -> None:
        first = spectral_to_xyz(self.material, self.light, "CIE1931_2")
        second = spectral_to_xyz(self.material, self.light, "CIE1964_10")
        self.assertNotEqual(first["observer"], second["observer"])
        self.assertNotEqual(first["cmf_hash"], second["cmf_hash"])
        self.assertGreater(
            max(abs(a - b) for a, b in zip(first["xyz"], second["xyz"])), 1e-5
        )

        stages_1931 = build_pipeline(self.scene, {}, revision=1).stages
        scene_1964_doc = copy.deepcopy(self.scene.to_dict())
        scene_1964_doc["scene"]["observer"] = "CIE1964_10"
        scene_1964 = parse_scene(scene_1964_doc)
        stages_1964 = build_pipeline(scene_1964, {}, revision=1).stages
        stage_id = spectral_stage_id("mat_oxide", "light_day")
        self.assertNotEqual(
            stages_1931[stage_id]["content_hash"],
            stages_1964[stage_id]["content_hash"],
        )
        self.assertNotEqual(
            stages_1931[adapted_stage_id("mat_oxide", "light_day")]["content_hash"],
            stages_1964[adapted_stage_id("mat_oxide", "light_day")]["content_hash"],
        )

    def test_bradford_matrix_maps_source_white_to_camera_white(self) -> None:
        source = (0.95047, 1.0, 1.08883)
        camera = (0.96422, 1.0, 0.82521)
        info = adaptation_matrix("bradford", source, camera)
        mapped = mat3_vec(info["matrix"], source)
        for actual, expected in zip(mapped, camera):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertTrue(info["applied"])
        self.assertEqual(info["basis"]["name"], "bradford")

    def test_von_kries_and_bradford_differ(self) -> None:
        source = (0.95047, 1.0, 1.08883)
        camera = (0.96422, 1.0, 0.82521)
        bradford = adaptation_matrix("bradford", source, camera)["matrix"]
        von_kries = adaptation_matrix("von_kries", source, camera)["matrix"]
        self.assertNotEqual(bradford, von_kries)

    def test_invalid_camera_white_point_blocks_downstream(self) -> None:
        spectral = spectral_to_xyz(self.material, self.light, "CIE1931_2")
        with self.assertRaises(DomainError) as context:
            adapt_xyz(spectral, [0.0, 1.0, 1.0], "bradford")
        self.assertEqual(context.exception.code, "E_WHITE_POINT")

    def test_target_colour_output(self) -> None:
        output = xyz_to_target([30.0, 32.0, 20.0], "sRGB")
        self.assertEqual(output["space"], "sRGB")
        self.assertEqual(len(output["rgb"]), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in output["rgb"]))
        xyz_output = xyz_to_target([30.0, 32.0, 20.0], "XYZ")
        self.assertIsNone(xyz_output["linear_rgb"])
        self.assertEqual(xyz_output["xyz"], [30.0, 32.0, 20.0])


if __name__ == "__main__":
    unittest.main()
