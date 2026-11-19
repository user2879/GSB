from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from story_workbench.canonical import pretty_json, read_utf8
from story_workbench.demo import build_demo_payload, run_demo
from story_workbench.samples import generate_samples


class DemoDeterminismTests(unittest.TestCase):
    def test_sample_generation_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_paths = generate_samples(first_dir)
            second_paths = generate_samples(second_dir)
            self.assertEqual([path.name for path in first_paths], [path.name for path in second_paths])
            for first, second in zip(first_paths, second_paths):
                self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_report_payload_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as sample_dir:
            generate_samples(sample_dir)
            first = pretty_json(build_demo_payload(sample_dir)).encode("utf-8")
            second = pretty_json(build_demo_payload(sample_dir)).encode("utf-8")
            self.assertEqual(first, second)

    def test_demo_artifacts_are_byte_identical_between_output_directories(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_summary = run_demo(first_dir)
            second_summary = run_demo(second_dir)
            self.assertTrue(first_summary["byte_identical"])
            self.assertTrue(second_summary["byte_identical"])
            self.assertEqual(first_summary["run_a_sha256"], second_summary["run_a_sha256"])
            for name in ("report.json", "node_hashes.json", "report.md"):
                self.assertEqual(
                    Path(first_dir, name).read_bytes(),
                    Path(second_dir, name).read_bytes(),
                    name,
                )
            report = read_utf8(Path(first_dir, "report.json"))
            self.assertIn('"byte_identical": true', report)
            self.assertIn('"multi_path_bad_status": "divergent"', report)


if __name__ == "__main__":
    unittest.main()
