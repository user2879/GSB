from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spectral_workbench.config import InjectionConfig
from spectral_workbench.errors import ErrorCode, WorkbenchError
from spectral_workbench.generator import generate_cube, load_manifest
from spectral_workbench.io_utils import hash_tree
from tests.helpers import small_cube_config


class GeneratorTests(unittest.TestCase):
    def test_same_configuration_is_byte_identical(self) -> None:
        config = small_cube_config()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            generated_first = generate_cube(config, InjectionConfig(), first)
            generated_second = generate_cube(config, InjectionConfig(), second)
            self.assertEqual(generated_first.content_hash, generated_second.content_hash)
            self.assertEqual(hash_tree(first), hash_tree(second))
            manifest = load_manifest(first, verify_hashes=True)
            self.assertEqual(manifest["shape"], [8, 32, 32])
            self.assertEqual(len(manifest["frame_stats"]), 8)
            for stats in manifest["frame_stats"]:
                self.assertIn("raw_peak", stats)
                self.assertIn("raw_mean", stats)
                self.assertIn("saturated_pixels", stats)
            truth = Path(first) / "truth/frame_0000.json"
            self.assertTrue(truth.is_file())

    def test_invalid_injection_is_rejected(self) -> None:
        config = small_cube_config()
        injection = InjectionConfig.from_dict(
            {"dark_inconsistency": {"offset_dn": 10.0, "band_start": 7, "band_end": 2}}
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(WorkbenchError) as context:
                generate_cube(config, injection, Path(temporary) / "cube")
            self.assertEqual(context.exception.code, ErrorCode.E_INJECT_CONFIG)

    def test_malformed_manifest_reports_cube_format_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text('{"shape": [2, 3]}', encoding="utf-8")
            with self.assertRaises(WorkbenchError) as context:
                load_manifest(root)
            self.assertEqual(context.exception.code, ErrorCode.E_CUBE_FORMAT)


if __name__ == "__main__":
    unittest.main()
