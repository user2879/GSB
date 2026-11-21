from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arbitration_workbench.config import parse_config
from arbitration_workbench.numeric import read_json
from arbitration_workbench.versions import (
    convert_file,
    read_v1,
    read_v2,
    read_v3,
    roundtrip_consistency,
    write_v1,
    write_v2,
    write_v3,
)

from .helpers import ROOT


class VersionCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = parse_config(read_json(ROOT / "samples" / "config.json"))
        cls.sample_dir = ROOT / "samples" / "version_samples"
        cls.records = read_v2(cls.sample_dir / "arbitration.v2.json", cls.config)

    def test_v2_json_roundtrip_and_float_precision(self):
        self.assertEqual(48, len(self.records))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.v2.json"
            write_v2(self.records, path, self.config, temp_dir=directory)
            restored = read_v2(path, self.config)
            check = roundtrip_consistency(self.records, restored)
            self.assertTrue(check["equal"])
            self.assertEqual(self.records, restored)

    def test_v2_to_v1_sidecar_restores_exact_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.v1.txt"
            info = write_v1(self.records, path, self.config, temp_dir=directory)
            sidecar = read_json(info["sidecar_path"])
            self.assertEqual(48, len(sidecar["records"]))
            self.assertTrue(sidecar["records"]["T0001"]["reason_codes"])
            self.assertIn("truncation_residual_fixed", sidecar["records"]["T0001"])
            restored = read_v1(path, self.config)
            self.assertEqual(self.records, restored)
            self.assertTrue(all(len(line) == info["v1_line_width"] for line in path.read_text().splitlines()))

    def test_v3_header_offsets_and_score_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.v3.bin"
            info = write_v3(self.records, path, self.config, temp_dir=directory)
            self.assertEqual(48, info["record_count"])
            self.assertGreater(info["score_block_offset"], info["offset_table_offset"])
            restored = read_v3(path, self.config)
            self.assertEqual(self.records, restored)
            self.assertTrue(roundtrip_consistency(self.records, restored)["equal"])

    def test_v3_to_v1_reports_lossy_details(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.v3.bin"
            target = Path(directory) / "target.v1.txt"
            write_v3(self.records, source, self.config, temp_dir=directory)
            report = convert_file(source, "v3", target, "v1", self.config, temp_dir=directory)
            self.assertEqual("v3", report["source_version"])
            self.assertEqual("v1", report["target_version"])
            self.assertIn("E_VERSION_LOSSY", report["codes"])
            self.assertEqual(48, len(report["degradation_details"]))
            fields = {field for item in report["degradation_details"] for field in item["fields"]}
            self.assertIn("reason_codes", fields)
            self.assertIn("score_truncation_residual", fields)
            self.assertTrue(Path(report["sidecar_path"]).exists())

    def test_triple_roundtrip_preserves_count_missing_and_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            v1 = Path(directory) / "x.v1"
            v3 = Path(directory) / "x.v3"
            v2 = Path(directory) / "x.v2"
            write_v1(self.records, v1, self.config, temp_dir=directory)
            from_v1 = read_v1(v1, self.config)
            write_v3(from_v1, v3, self.config, temp_dir=directory)
            from_v3 = read_v3(v3, self.config)
            write_v2(from_v3, v2, self.config, temp_dir=directory)
            from_v2 = read_v2(v2, self.config)
            self.assertTrue(roundtrip_consistency(self.records, from_v1)["equal"])
            self.assertTrue(roundtrip_consistency(self.records, from_v3)["equal"])
            self.assertTrue(roundtrip_consistency(self.records, from_v2)["equal"])
            self.assertEqual(self.records, from_v2)


if __name__ == "__main__":
    unittest.main()

