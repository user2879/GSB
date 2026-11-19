from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from spectral_workspace.canonical import atomic_write_json, read_json
from spectral_workspace.errors import DomainError
from spectral_workspace.model import load_scene_json, parse_scene, write_scene_json
from spectral_workspace.synth import generate_sample_scene


class ModelValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = generate_sample_scene()

    def test_valid_scene_and_deterministic_json_roundtrip(self) -> None:
        scene = parse_scene(self.sample)
        self.assertEqual(len(scene.materials), 3)
        self.assertEqual(len(scene.lights), 2)
        self.assertEqual(len(scene.materials["mat_oxide"].reflectance), 81)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            atomic_write_json(first, self.sample, pretty=True)
            parsed = parse_scene(read_json(first))
            atomic_write_json(second, parsed.to_dict(), pretty=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_explicit_scene_file_api(self) -> None:
        scene = parse_scene(self.sample)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            write_scene_json(path, scene)
            loaded = load_scene_json(path)
            self.assertEqual(loaded.to_dict(), scene.to_dict())

    def test_reflectance_out_of_range_has_raw_pointer(self) -> None:
        payload = copy.deepcopy(self.sample)
        payload["materials"]["mat_oxide"]["reflectance"][17] = 1.25
        with self.assertRaises(DomainError) as context:
            parse_scene(payload)
        error = context.exception
        self.assertEqual(error.code, "E_SPECTRUM_RANGE")
        self.assertEqual(error.pointer, "/materials/mat_oxide/reflectance/17")
        self.assertEqual(error.details["raw"], 1.25)
        self.assertEqual(error.details["wavelength_nm"], 465)

    def test_missing_sample_is_rejected(self) -> None:
        payload = copy.deepcopy(self.sample)
        payload["materials"]["mat_oxide"]["reflectance"].pop()
        with self.assertRaises(DomainError) as context:
            parse_scene(payload)
        self.assertEqual(context.exception.code, "E_SPECTRUM_RANGE")
        self.assertEqual(
            context.exception.pointer, "/materials/mat_oxide/reflectance"
        )
        self.assertEqual(context.exception.details["actual_count"], 80)

    def test_non_monotonic_wavelength_is_rejected(self) -> None:
        payload = copy.deepcopy(self.sample)
        wavelengths = payload["materials"]["mat_oxide"]["wavelengths_nm"]
        wavelengths[20] = wavelengths[19]
        with self.assertRaises(DomainError) as context:
            parse_scene(payload)
        self.assertEqual(context.exception.code, "E_SPECTRUM_RANGE")
        self.assertEqual(
            context.exception.pointer,
            "/materials/mat_oxide/wavelengths_nm/20",
        )

    def test_observer_invalid(self) -> None:
        payload = copy.deepcopy(self.sample)
        payload["scene"]["observer"] = "CIE2099_2"
        with self.assertRaises(DomainError) as context:
            parse_scene(payload)
        self.assertEqual(context.exception.code, "E_OBSERVER_INVALID")
        self.assertEqual(context.exception.pointer, "/scene/observer")

    def test_white_point_zero(self) -> None:
        payload = copy.deepcopy(self.sample)
        payload["scene"]["camera_white_point_xyz"] = [0.0, 1.0, 1.08883]
        with self.assertRaises(DomainError) as context:
            parse_scene(payload)
        self.assertEqual(context.exception.code, "E_WHITE_POINT")
        self.assertEqual(
            context.exception.pointer, "/scene/camera_white_point_xyz/0"
        )


if __name__ == "__main__":
    unittest.main()
