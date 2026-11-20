import json
import os
import shutil
import tempfile
import unittest

from sft_diag.errors import E_CONFIG_MISMATCH, OK
from sft_diag.runner import compare_artifacts, run_workbench
from tests._helpers import minimal_sample, write_json


BATCHES = [
    {
        "batch_no": 1,
        "operations": [
            {
                "op": "replace_template",
                "seed": {
                    "seed_id": "seed-1",
                    "task_type": "alpha",
                    "template_id": "tpl-a0",
                    "template": "Categorize the {subject} memo covering {topic} now.",
                    "slots": ["subject", "topic"],
                    "answer_template": "answer:{subject}:{topic}",
                },
            },
            {"op": "set_target_count", "task_type": "alpha", "value": 14},
        ],
    },
    {
        "batch_no": 2,
        "operations": [
            {
                "op": "add_slot_source",
                "source": {
                    "source_id": "src-9", "task_type": "alpha", "slot_name": "topic",
                    "values": ["botany", "geology", "astronomy"],
                },
            },
            {"op": "set_slot_weight", "task_type": "alpha",
             "source_id": "src-9", "value": 2.0},
        ],
    },
    {
        "batch_no": 3,
        "operations": [
            {"op": "delete_template", "seed_id": "seed-2"},
            {"op": "set_score_weight", "dimension": "clarity", "value": 0.45},
        ],
    },
    {
        "batch_no": 4,
        "operations": [
            {"op": "delete_slot_source", "source_id": "src-2"},
            {"op": "set_dedup_threshold", "value": 0.42},
        ],
    },
    {
        "batch_no": 5,
        "operations": [
            {"op": "set_boundary_rule", "value": "split"},
            {"op": "set_template_weight", "task_type": "alpha",
             "template_id": "tpl-a0", "value": 5.0},
        ],
    },
]


class IncrementalConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.sample_dir = minimal_sample(
            self.root, threshold=0.5, interval=2, extra_batches=BATCHES
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _full_build(self, up_to_batch: int) -> dict:
        output_dir = os.path.join(self.root, f"full-{up_to_batch}")
        result = run_workbench(
            self.sample_dir, output_dir,
            up_to_batch=up_to_batch, shadow_check=False,
        )
        self.assertEqual(result.business_code, OK)
        return result.stage_artifacts

    def test_every_batch_matches_full_rebuild_field_by_field(self) -> None:
        incremental_dir = os.path.join(self.root, "inc")
        for batch_no in range(1, 6):
            result = run_workbench(
                self.sample_dir, incremental_dir,
                up_to_batch=batch_no, shadow_check=False,
            )
            self.assertEqual(result.business_code, OK, result.failures)
            full = self._full_build(batch_no)
            differences = compare_artifacts(
                result.stage_artifacts, full, source_batch=batch_no
            )
            self.assertEqual(differences, [], f"batch {batch_no} differs: {differences[:3]}")

    def test_difference_records_carry_source_batch(self) -> None:
        from sft_diag.runner import _diff_objects
        differences = _diff_objects(
            "scoring", {"a": 1}, {"a": 2}, "$", source_batch=7
        )
        self.assertEqual(differences[0]["source_batch"], 7)

    def test_built_in_shadow_diagnostic_reports_match(self) -> None:
        result = run_workbench(
            self.sample_dir,
            os.path.join(self.root, "shadow-inc"),
            up_to_batch=5,
            shadow_check=True,
        )
        self.assertTrue(result.report["diagnostics"]["shadow_match"])
        self.assertEqual(result.report["diagnostics"]["field_differences"], [])

    def test_invalidation_and_reuse_counts(self) -> None:
        output_dir = os.path.join(self.root, "reuse")
        first = run_workbench(
            self.sample_dir, output_dir, up_to_batch=1, shadow_check=False
        )
        invalidation = first.report["invalidation"]
        self.assertEqual(len(invalidation["new_candidates"]), 14)
        self.assertEqual(invalidation["sampling_reused_candidates"], [])
        second = run_workbench(
            self.sample_dir, output_dir, up_to_batch=2, shadow_check=False
        )
        invalidation = second.report["invalidation"]
        self.assertEqual(invalidation["new_candidates"], [])
        self.assertTrue(invalidation["sampling_changed_candidates"])
        rescored = set(invalidation["rescored_candidates"])
        reused = set(invalidation["score_reused_candidates"])
        all_ids = {c["candidate_id"] for c in
                   second.stage_artifacts["sampling"]["candidates"]}
        self.assertTrue(rescored)
        self.assertTrue(reused)
        self.assertEqual(rescored | reused, all_ids)
        self.assertTrue(rescored.isdisjoint(reused))
        self.assertTrue(invalidation["reasons"]["target_or_weight_changed"])
        self.assertIn(
            "src-9",
            second.stage_artifacts["sampling"]["reference_index"]["slot_source"],
        )

        third = run_workbench(
            self.sample_dir, output_dir, up_to_batch=3, shadow_check=False
        )
        self.assertEqual(
            set(third.report["invalidation"]["rescored_candidates"]), all_ids
        )
        # 输入完全不变的重跑：所有打分结果必须直接复用，不得重新打分。
        rerun = run_workbench(
            self.sample_dir, output_dir, up_to_batch=3, shadow_check=False
        )
        self.assertEqual(rerun.report["invalidation"]["rescored_candidates"], [])
        self.assertEqual(
            set(rerun.report["invalidation"]["score_reused_candidates"]), all_ids
        )

    def test_resume_after_full_completion_is_idempotent(self) -> None:
        output_dir = os.path.join(self.root, "resume")
        first = run_workbench(
            self.sample_dir, output_dir, up_to_batch=3, shadow_check=False
        )
        second = run_workbench(
            self.sample_dir, output_dir, up_to_batch=3, shadow_check=False
        )
        self.assertEqual(second.recovery_batch_no, 3)
        self.assertEqual(
            first.stage_artifacts["scoring"], second.stage_artifacts["scoring"]
        )
        self.assertEqual(
            second.report["invalidation"]["rescored_candidates"], []
        )

    def test_config_mismatch_rejects_resume(self) -> None:
        output_dir = os.path.join(self.root, "mismatch")
        run_workbench(
            self.sample_dir, output_dir, up_to_batch=1, shadow_check=False
        )
        other_dir = os.path.join(self.root, "other-sample")
        shutil.copytree(self.sample_dir, other_dir)
        config_path = os.path.join(other_dir, "build_config.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        config["max_workers"] = 2
        write_json(config_path, config)
        result = run_workbench(
            other_dir, output_dir, up_to_batch=1, shadow_check=False
        )
        self.assertEqual(result.business_code, E_CONFIG_MISMATCH)
        self.assertTrue(any(f.code == E_CONFIG_MISMATCH for f in result.failures))


if __name__ == "__main__":
    unittest.main()
