import json
import os
import shutil
import tempfile
import unittest

from mvrec.cli import run_pipeline
from mvrec.samplegen import generate_sample


class ConcurrencyMatrixTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="mvrec_matrix_")
        self.sample_dir = os.path.join(self.workdir, "sample")
        generate_sample(self.sample_dir, 8)
        base = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "recon_config.json"),
                              encoding="utf-8"))
        self.base = base

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_all_concurrency_levels_produce_identical_artifacts(self):
        reference = None
        for thread_count in (1, 2, 3, 5, 8):
            config = dict(self.base, thread_count=thread_count)
            config_path = os.path.join(self.workdir, "cfg_%d.json" % thread_count)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            snap = os.path.join(self.workdir, "snap_%d" % thread_count)
            report, _ = run_pipeline(
                self.sample_dir, config_path, None, snap,
                os.path.join(self.workdir, "report_%d.json" % thread_count),
            )
            cloud_path = os.path.join(
                snap, "merge_v000001", "point_cloud.json")
            with open(cloud_path, encoding="utf-8") as handle:
                cloud = json.load(handle)
            fingerprint = {
                "cloud": cloud,
                "deduction": report["deduction_sequence"],
                "curve": report["budget_curve"],
                "failures": report["failures"],
                "pending": report["pending_queue"],
                "depth_summaries": self._load_depth_summaries(snap),
            }
            if reference is None:
                reference = fingerprint
            else:
                self.assertEqual(fingerprint, reference)

    @staticmethod
    def _load_depth_summaries(snap):
        matching = sorted(name for name in os.listdir(snap)
                          if name.startswith("depth_v"))[-1]
        with open(os.path.join(snap, matching, "depth_summaries.json"),
                  encoding="utf-8") as handle:
            return json.load(handle)


if __name__ == "__main__":
    unittest.main()
