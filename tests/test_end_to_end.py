import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.generate_samples import write_bundle
from texmat.testkit import run_incremental


ROOT = Path(__file__).resolve().parents[1]


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sample = self.tmp / "sample"
        write_bundle(self.sample, size=18, workers=4, budget=100000)

    def tearDown(self):
        self._tmp.cleanup()

    def _cli(self, snapshot: Path, report: Path | None = None, extra=None):
        cmd = [sys.executable, "-m", "texmat", "run",
               "--sample-dir", str(self.sample),
               "--snapshot-dir", str(snapshot)]
        if report is not None:
            cmd += ["--report", str(report)]
        if extra:
            cmd += extra
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_cli_full_run_and_report(self):
        snapshot = self.tmp / "snap"
        report = self.tmp / "report.json"
        proc = self._cli(snapshot, report)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["status"], "complete")
        self.assertIn("error_attribution", payload)
        self.assertIn("stages", payload["error_attribution"])
        self.assertIn("cache", payload)
        self.assertIn("budget", payload)
        self.assertIn("failures", payload)
        # PNG outputs exist and are valid RGBA images
        for material in ("mat_stone", "mat_metal"):
            png = snapshot / "png" / f"{material}.png"
            self.assertTrue(png.exists())
            with Image.open(png) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.size, (18, 18))

    def test_quantize_off_changes_pipeline_end_to_end(self):
        on_snap = self.tmp / "on"
        run_incremental(self.sample, on_snap)
        raw = json.loads((self.sample / "generation_config.json").read_text())
        raw["quantize"] = False
        (self.sample / "generation_config.json").write_text(json.dumps(raw, sort_keys=True))
        off_snap = self.tmp / "off"
        run_incremental(self.sample, off_snap)
        on_data = np.load(on_snap / "layers" / "layer01.npz")
        off_data = np.load(off_snap / "layers" / "layer01.npz")
        self.assertTrue(on_data["quantized"].size > 0)
        self.assertEqual(off_data["quantized"].size, 0)
        self.assertFalse(np.array_equal(on_data["accum"], off_data["accum"]))

    def test_budget_resume_reaches_completion(self):
        raw = json.loads((self.sample / "generation_config.json").read_text())
        raw["point_budget"] = 200
        (self.sample / "generation_config.json").write_text(json.dumps(raw, sort_keys=True))
        snapshot = self.tmp / "snap"
        report = self.tmp / "report.json"
        proc = self._cli(snapshot, report)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["status"], "complete")
        self.assertGreater(payload["batches"], 1)
        # every budget event carries an order key and a known action
        for event in payload["budget"]["events"]:
            self.assertIn(event["action"], ("admitted", "pending", "admitted_oversize"))

    def test_corrupt_texture_does_not_abort_run(self):
        # Add a corrupt PNG alongside the good ones; run still completes.
        (self.sample / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
        snapshot = self.tmp / "snap"
        report = self.tmp / "report.json"
        proc = self._cli(snapshot, report)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(report.read_text())
        codes = {item["code"] for item in payload["failures"]}
        self.assertIn("E_TEXTURE_DECODE", codes)
        self.assertEqual(payload["status"], "complete")

    def test_invalid_config_emits_code_and_nonzero_exit(self):
        raw = json.loads((self.sample / "generation_config.json").read_text())
        raw["resolution_tiers"] = [32, 18]
        (self.sample / "generation_config.json").write_text(json.dumps(raw, sort_keys=True))
        report = self.tmp / "bad_report.json"
        proc = self._cli(self.tmp / "snap", report)
        self.assertNotEqual(proc.returncode, 0)
        payload = json.loads(report.read_text())
        self.assertEqual(payload["code"], "E_CONFIG_INVALID")


if __name__ == "__main__":
    unittest.main()
