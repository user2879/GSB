from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stemscope.conversion import convert_package
from stemscope.injection import load_injection_manifest, run_injection_case
from stemscope.model import FORMAT_VERSIONS
from stemscope.package import parse_package
from stemscope.samplegen import DEFAULT_SEED, generate_samples


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name) / "sample_data"
        generate_samples(cls.root, seed=DEFAULT_SEED)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_three_formats_are_frame_identical_and_repeatable(self) -> None:
        segment = self.root / "segments" / "seg_000"
        results = {
            version: parse_package(segment, version, clip_id="seg_000")
            for version in FORMAT_VERSIONS
        }
        self.assertTrue(all(item.ok for item in results.values()))
        packages = [results[version].package for version in FORMAT_VERSIONS]
        self.assertTrue(all(package is not None for package in packages))
        names = sorted(packages[0].tracks)
        for package in packages[1:]:
            self.assertEqual(sorted(package.tracks), names)
            for name in names:
                self.assertTrue(np.array_equal(packages[0].tracks[name].pcm, package.tracks[name].pcm))
        repeated = parse_package(segment, "v1", clip_id="seg_000")
        self.assertEqual(
            repeated.package.package_fingerprint,
            results["v1"].package.package_fingerprint,
        )
        for name in names:
            self.assertTrue(
                np.array_equal(repeated.package.tracks[name].pcm, results["v1"].package.tracks[name].pcm)
            )

    def test_conversion_matrix_preserves_pcm_and_checksum_loss(self) -> None:
        segment = self.root / "segments" / "seg_000"
        parsed = {version: parse_package(segment, version).package for version in FORMAT_VERSIONS}
        with tempfile.TemporaryDirectory() as temp:
            for source_version in FORMAT_VERSIONS:
                for target_version in FORMAT_VERSIONS:
                    if source_version == target_version:
                        continue
                    result = convert_package(
                        parsed[source_version],
                        target_version,
                        Path(temp) / f"{source_version}_to_{target_version}",
                    )
                    self.assertTrue(result.ok, result.failures)
                    for name in parsed[source_version].tracks:
                        self.assertTrue(
                            np.array_equal(
                                parsed[source_version].tracks[name].pcm,
                                result.target_package.tracks[name].pcm,
                            )
                        )
                    if source_version == "v3" and target_version == "v2":
                        self.assertTrue(
                            any(
                                loss["field"] == "block_checksum_crc32"
                                and loss["source_track"] == name
                                for loss in result.record["losses"]
                                for name in parsed[source_version].tracks
                            )
                        )
                        self.assertTrue(result.target_package.sidecar_bytes)

    def test_sidecar_participates_in_package_fingerprint(self) -> None:
        segment = self.root / "segments" / "seg_000"
        source = parse_package(segment, "v3").package
        with tempfile.TemporaryDirectory() as temp:
            result = convert_package(source, "v2", Path(temp))
            self.assertTrue(result.ok)
            sidecar_path = Path(temp) / "packages" / "v2" / "manifest.sidecar.json"
            before = parse_package(Path(temp), "v2").package.package_fingerprint
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["test_marker"] = "fingerprint-sensitive"
            sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
            after = parse_package(Path(temp), "v2").package.package_fingerprint
            self.assertNotEqual(before, after)

    def test_injection_cases_fail_with_parse_code(self) -> None:
        manifest = load_injection_manifest(self.root / "injection_manifest.json")
        with tempfile.TemporaryDirectory() as temp:
            for case in manifest["cases"]:
                result = run_injection_case(
                    self.root,
                    Path(temp) / case["id"],
                    case,
                )
                self.assertTrue(result["passed"], result)
                self.assertIn("E_TRACK_PARSE_FAILED", result["observed_codes"])


if __name__ == "__main__":
    unittest.main()
