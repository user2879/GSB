from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pointcloud_studio.errors import ErrorCode, StudioError
from pointcloud_studio.formats import load_normals, load_points
from pointcloud_studio.sample import SampleConfig, load_sample_bundle, write_sample_bundle


class SampleAndFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="pcstudio-sample-tests-")
        cls.root = Path(cls.temporary.name)
        cls.first = cls.root / "first"
        cls.second = cls.root / "second"
        cls.manifest = write_sample_bundle(cls.first, SampleConfig())
        write_sample_bundle(cls.second, SampleConfig())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_bundle_is_byte_identical(self) -> None:
        for name in ("points.bin", "normals.bin", "camera.json", "edits.json", "manifest.json"):
            self.assertEqual((self.first / name).read_bytes(), (self.second / name).read_bytes(), name)

    def test_manifest_and_arrays(self) -> None:
        self.assertEqual(self.manifest["point_count"], 120_000)
        self.assertEqual(sum(self.manifest["normal_confidence"]["histogram"]["counts"]), 120_000)
        self.assertEqual(self.manifest["class_counts"], {"cube_room": 72_000, "ground": 48_000})
        points = load_points(self.first / "points.bin")
        normals, confidence = load_normals(self.first / "normals.bin")
        self.assertEqual(points.shape, (120_000, 3))
        self.assertEqual(normals.shape, (120_000, 3))
        np.testing.assert_allclose(np.linalg.norm(normals.astype(np.float64), axis=1), 1.0, atol=2e-5)
        self.assertTrue(((confidence >= 0.0) & (confidence <= 1.0)).all())
        expected_min = np.asarray(self.manifest["bounds"]["min"])
        expected_max = np.asarray(self.manifest["bounds"]["max"])
        np.testing.assert_allclose(points.min(axis=0), expected_min, atol=1e-8)
        np.testing.assert_allclose(points.max(axis=0), expected_max, atol=1e-8)

    def test_invalid_magic_returns_business_error(self) -> None:
        corrupt = self.root / "corrupt"
        corrupt.mkdir()
        for path in self.first.iterdir():
            (corrupt / path.name).write_bytes(path.read_bytes())
        payload = bytearray((corrupt / "points.bin").read_bytes())
        payload[0] ^= 0xFF
        (corrupt / "points.bin").write_bytes(payload)
        with self.assertRaises(StudioError) as caught:
            load_sample_bundle(corrupt)
        self.assertEqual(caught.exception.code, ErrorCode.E_POINT_CLOUD_FORMAT)
        self.assertIn("path", caught.exception.context)


if __name__ == "__main__":
    unittest.main()

