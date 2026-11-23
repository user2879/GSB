import json
import tempfile
import unittest
from pathlib import Path

from tests._util import ROOT, make_workspace, run
from mstc.injection import INJECTION_CODE, load_injections, run_injection_suite


class BoundaryDiagnosticTests(unittest.TestCase):
    def test_boundary_records_have_reference_and_fixed(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        report, _ = run(ws)
        self.assertTrue(report.boundary_records)
        rec = report.boundary_records[0]
        self.assertIn(rec.difference_direction, (
            "fixed_underestimates_reference_class",
            "fixed_overestimates_winner_class",
            "equal",
        ))
        self.assertIsInstance(rec.reference_margin, float)
        self.assertIsInstance(rec.fixed_margin, int)

    def test_flip_code_uses_half_step(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        report, _ = run(ws)
        for rec in report.boundary_records:
            self.assertLessEqual(rec.fixed_margin, rec.half_step)


class InjectionSuiteTests(unittest.TestCase):
    def test_all_injections_map_to_business_codes(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        import shutil
        shutil.copy(ROOT / "configs" / "injections.json", ws / "injections.json")
        injections = load_injections(ws / "injections.json")
        doc = run_injection_suite(
            ws, "config.yaml", "samples\\manifest.json", injections,
            (3, 3), 16, 4, ws / "reports" / "inj.json",
        )
        self.assertEqual(doc["summary"]["total"], 6)
        self.assertEqual(doc["summary"]["hit"], 6)
        for entry in doc["injections"]:
            self.assertTrue(entry["injection_id"])
            self.assertEqual(entry["expected_code"], INJECTION_CODE[entry["kind"]])
            # a single failed tile must not wipe out the other tiles
            if entry["tile_id"]:
                self.assertGreaterEqual(len(entry["other_tiles_still_classified"]), 7)


if __name__ == "__main__":
    unittest.main()
