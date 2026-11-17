import json
import os
import shutil
import tempfile
import unittest

from mvrec.cli import run_pipeline
from mvrec.samplegen import generate_sample


class PendingQueueTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="mvrec_pending_")
        self.sample_dir = os.path.join(self.workdir, "sample")
        generate_sample(self.sample_dir, 8)
        base = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "recon_config.json"),
                              encoding="utf-8"))
        self.base = base

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def test_suspension_retains_every_unfinished_group_in_stable_order(self):
        config = dict(self.base, point_budget=60, thread_count=3)
        config_path = os.path.join(self.workdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        snap = os.path.join(self.workdir, "snap")
        report, _ = run_pipeline(
            self.sample_dir, config_path, None, snap,
            os.path.join(self.workdir, "report.json"),
        )
        self.assertIn("E_POINT_BUDGET_EXHAUSTED",
                      {f["code"] for f in report["failures"]})
        self.assertEqual(report["pending_queue"], sorted(report["pending_queue"]))
        manifest_path = os.path.join(snap, "manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["pending_queue"], report["pending_queue"])
        for group_index in report["pending_queue"]:
            self.assertEqual(
                manifest["group_progress"][str(group_index)]["status"],
                "suspended",
            )

        # Resume with a large budget and confirm the pending groups all commit.
        resume_config = dict(self.base, point_budget=100000, thread_count=5)
        resume_path = os.path.join(self.workdir, "resume_config.json")
        with open(resume_path, "w", encoding="utf-8") as handle:
            json.dump(resume_config, handle)
        resumed, _ = run_pipeline(
            self.sample_dir, resume_path, None, snap,
            os.path.join(self.workdir, "resume_report.json"),
        )
        self.assertEqual(resumed["pending_queue"], [])
        self.assertEqual(
            resumed["resume_point"]["committed_groups"],
            sorted(resumed["resume_point"]["committed_groups"]),
        )
        self.assertGreater(resumed["point_cloud_size"], 0)


if __name__ == "__main__":
    unittest.main()
