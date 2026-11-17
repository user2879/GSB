import json
import os
import shutil
import tempfile
import unittest

from mvrec.cli import run_pipeline
from mvrec.samplegen import generate_sample


class PipelineTestBase(unittest.TestCase):
    VIEW_COUNT = 8

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="mvrec_test_")
        self.sample_dir = os.path.join(self.workdir, "sample")
        generate_sample(self.sample_dir, self.VIEW_COUNT)
        self.config_path = os.path.join(self.workdir, "config.json")
        with open(os.path.join(os.path.dirname(__file__), "..", "configs",
                               "recon_config.json"), encoding="utf-8") as handle:
            self.base_config = json.load(handle)

    def tearDown(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def write_config(self, **overrides):
        config = dict(self.base_config)
        config.update(overrides)
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        return self.config_path

    def execute(self, config_path, snapshot=None, injection_path=None):
        report_path = os.path.join(self.workdir, "report_%d.json" % os.getpid())
        report, pipeline = run_pipeline(
            self.sample_dir, config_path, injection_path,
            snapshot or os.path.join(self.workdir, "snap"),
            report_path,
        )
        return report, pipeline, report_path


class DeterminismTests(PipelineTestBase):
    def test_serial_and_parallel_point_clouds_are_identical(self):
        serial_path = self.write_config(thread_count=1)
        parallel_path = self.write_config(thread_count=4)
        serial_report, _, _ = self.execute(serial_path, os.path.join(self.workdir, "s1"))
        parallel_report, _, _ = self.execute(parallel_path, os.path.join(self.workdir, "s2"))
        self.assertEqual(serial_report["point_cloud_size"], parallel_report["point_cloud_size"])
        self.assertGreater(serial_report["point_cloud_size"], 0)
        self.assertEqual(
            serial_report["grouping_decision"], parallel_report["grouping_decision"]
        )
        self.assertEqual(
            serial_report["deduction_sequence"], parallel_report["deduction_sequence"]
        )
        self.assertEqual(serial_report["budget_curve"], parallel_report["budget_curve"])
        with open(os.path.join(self.workdir, "s1", "merge_v000001",
                               "point_cloud.json"), encoding="utf-8") as handle:
            serial_cloud = json.load(handle)
        with open(os.path.join(self.workdir, "s2", "merge_v000001",
                               "point_cloud.json"), encoding="utf-8") as handle:
            parallel_cloud = json.load(handle)
        self.assertEqual(serial_cloud, parallel_cloud)

    def test_decision_record_ignores_concurrency(self):
        path_a = self.write_config(thread_count=1)
        path_b = self.write_config(thread_count=8)
        report_a, _, _ = self.execute(path_a, os.path.join(self.workdir, "da"))
        report_b, _, _ = self.execute(path_b, os.path.join(self.workdir, "db"))
        self.assertEqual(
            report_a["grouping_decision"]["candidate_scores"],
            report_b["grouping_decision"]["candidate_scores"],
        )
        self.assertEqual(
            report_a["grouping_decision"]["selected"],
            report_b["grouping_decision"]["selected"],
        )


class BudgetTests(PipelineTestBase):
    def test_point_budget_suspends_with_stable_pending_queue(self):
        path = self.write_config(point_budget=120, thread_count=3)
        report, _, _ = self.execute(path, os.path.join(self.workdir, "snap"))
        codes = {failure["code"] for failure in report["failures"]}
        self.assertIn("E_POINT_BUDGET_EXHAUSTED", codes)
        self.assertEqual(report["pending_queue"], sorted(report["pending_queue"]))
        self.assertTrue(all(
            consumed <= 120 for consumed in
            [point["consumed_points"] for point in report["budget_curve"]]
        ))

    def test_resume_after_budget_increase_completes_all_groups(self):
        first_path = self.write_config(point_budget=120, thread_count=2)
        snap = os.path.join(self.workdir, "snap_resume")
        first_report, _, _ = self.execute(first_path, snap)
        self.assertIn("E_POINT_BUDGET_EXHAUSTED",
                      {f["code"] for f in first_report["failures"]})
        second_path = self.write_config(point_budget=100000, thread_count=4)
        second_report, _, _ = self.execute(second_path, snap)
        self.assertNotIn(
            "E_POINT_BUDGET_EXHAUSTED",
            {f["code"] for f in second_report["failures"]},
        )
        self.assertEqual(second_report["pending_queue"], [])
        self.assertGreater(second_report["point_cloud_size"], 0)
        self.assertTrue(second_report["resume_point"]["committed_groups"])

    def test_zero_slot_budget_maps_to_business_code(self):
        path = self.write_config(slot_budget=0)
        report, _, _ = self.execute(path, os.path.join(self.workdir, "snap_slot"))
        codes = {failure["code"] for failure in report["failures"]}
        self.assertIn("E_SLOT_BUDGET_EXHAUSTED", codes)
        self.assertGreater(len(report["pending_queue"]), 0)
        self.assertEqual(report["pending_queue"], sorted(report["pending_queue"]))

    def test_zero_point_budget_maps_to_business_code(self):
        path = self.write_config(point_budget=0)
        report, _, _ = self.execute(path, os.path.join(self.workdir, "snap_point"))
        codes = {failure["code"] for failure in report["failures"]}
        self.assertIn("E_POINT_BUDGET_EXHAUSTED", codes)
        self.assertGreater(len(report["pending_queue"]), 0)


if __name__ == "__main__":
    unittest.main()
