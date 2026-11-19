import json
import os
import tempfile
import unittest

import soundfile as sf

from voice_repair_workbench import synthesis
from voice_repair_workbench.engine import RepairEngine
from voice_repair_workbench.metrics import compare_conclusions, segment_metrics
from voice_repair_workbench.persistence import Workspace
from voice_repair_workbench.project import default_v2_project


class TestOutputAndMetrics(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.paths = synthesis.generate_material(os.path.join(self.work, "m"))
        with open(self.paths["segments"], encoding="utf-8") as fh:
            self.segments = json.load(fh)["segments"]

    def test_output_bit_depth_policy(self):
        for depth, subtype in ((16, "PCM_16"), (24, "PCM_24")):
            proj = default_v2_project()
            proj["bit_depth_policy"] = {
                "output_bit_depth": depth,
                "mode": f"force{depth}",
            }
            ws = Workspace(os.path.join(self.work, f"ws{depth}"))
            RepairEngine(ws).execute(
                self.paths["mono16"], proj, self.segments, expected_channels=1
            )
            info = sf.info(os.path.join(ws.root, "repaired.wav"))
            self.assertEqual(info.subtype, subtype)

    def test_segment_metrics_shape_and_tolerance_constant(self):
        import numpy as np

        signal = np.zeros(self.segments[-1]["end_sample"])
        rows = segment_metrics(signal, self.segments)
        self.assertEqual(len(rows), len(self.segments))
        self.assertIn("rms_db", rows[0])
        verdict = compare_conclusions(rows, rows)
        self.assertTrue(verdict["consistent"])
        self.assertIn("rms_abs", verdict["tolerance"])


if __name__ == "__main__":
    unittest.main()
