import os
import tempfile
import unittest

import json

from sft_diag.runner import run_workbench
from tests._helpers import minimal_sample


class ConcurrencyOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_worker_count_does_not_change_output(self) -> None:
        outputs = {}
        sample_dir = minimal_sample(os.path.join(self.root, "sample-base"))
        config_path = os.path.join(sample_dir, "build_config.json")
        with open(config_path, encoding="utf-8") as handle:
            base_config = json.load(handle)
        for workers in (1, 2, 4, 8):
            config = dict(base_config)
            config["max_workers"] = workers
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, sort_keys=True)
            result = run_workbench(
                sample_dir,
                os.path.join(self.root, f"out-{workers}"),
                shadow_check=False,
            )
            outputs[workers] = result.stage_artifacts
        baseline = outputs[1]
        for workers in (2, 4, 8):
            self.assertEqual(outputs[workers]["sampling"], baseline["sampling"])
            self.assertEqual(outputs[workers]["scoring"], baseline["scoring"])
            self.assertEqual(outputs[workers]["dedup"], baseline["dedup"])

    def test_report_ordering_is_identifier_sorted(self) -> None:
        sample_dir = minimal_sample(os.path.join(self.root, "sample"))
        result = run_workbench(
            sample_dir, os.path.join(self.root, "out"), shadow_check=False
        )
        score_ids = [item["candidate_id"] for item in
                     result.stage_artifacts["scoring"]["scores"]]
        self.assertEqual(score_ids, sorted(score_ids))
        for task_result in result.stage_artifacts["dedup"]["results"]:
            for group in task_result["groups"]:
                self.assertEqual(group["members"], sorted(group["members"]))
        failure_ids = [failure["ref"] for failure in result.failures]
        self.assertEqual(failure_ids, sorted(failure_ids))


if __name__ == "__main__":
    unittest.main()
