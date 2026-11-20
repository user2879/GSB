import json
import os
import tempfile
import unittest

from sft_diag.errors import OK
from sft_diag.runner import run_workbench
from sft_diag.snapshot import SnapshotStore
from tests._helpers import minimal_sample


BATCHES = [
    {"batch_no": 1, "operations": [
        {"op": "set_target_count", "task_type": "alpha", "value": 12}]},
    {"batch_no": 2, "operations": [
        {"op": "set_score_weight", "dimension": "format", "value": 0.33}]},
    {"batch_no": 3, "operations": [
        {"op": "set_dedup_threshold", "value": 0.35}]},
]


class CrashRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.sample_dir = minimal_sample(
            self.root, threshold=0.5, interval=1, extra_batches=BATCHES
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _crash_hook(self, stage: str, batch_no: int) -> None:
        if stage == self.crash_stage and batch_no == self.crash_batch:
            raise KeyboardInterrupt("simulated forced termination")

    def _run_with_crash(self, crash_stage: str, crash_batch: int) -> None:
        self.crash_stage = crash_stage
        self.crash_batch = crash_batch
        output_dir = os.path.join(self.root, f"crash-{crash_stage}-{crash_batch}")
        with self.assertRaises(KeyboardInterrupt):
            run_workbench(
                self.sample_dir, output_dir,
                up_to_batch=crash_batch, shadow_check=False,
                fault_hook=self._crash_hook,
            )
        store = SnapshotStore(os.path.join(output_dir, "snapshot"))
        leftovers = [name for name in os.listdir(store.root)
                     if name.startswith("tmp.")]
        self.assertEqual(leftovers, [])
        recovered = run_workbench(
            self.sample_dir, output_dir,
            up_to_batch=crash_batch, shadow_check=False,
        )
        self.assertEqual(recovered.business_code, OK)
        full_dir = os.path.join(self.root, f"full-{crash_stage}-{crash_batch}")
        full = run_workbench(
            self.sample_dir, full_dir,
            up_to_batch=crash_batch, shadow_check=False,
        )
        self.assertEqual(
            recovered.stage_artifacts["sampling"], full.stage_artifacts["sampling"]
        )
        self.assertEqual(
            recovered.stage_artifacts["scoring"], full.stage_artifacts["scoring"]
        )
        self.assertEqual(
            recovered.stage_artifacts["dedup"], full.stage_artifacts["dedup"]
        )

    def test_crash_before_each_stage_commit(self) -> None:
        for stage in ("sampling_pre", "scoring_pre", "dedup_pre", "sampling_post"):
            with self.subTest(stage=stage):
                self._run_with_crash(stage, 2)

    def test_recovery_point_batch_recorded(self) -> None:
        output_dir = os.path.join(self.root, "points")
        run_workbench(
            self.sample_dir, output_dir, up_to_batch=2, shadow_check=False
        )
        result = run_workbench(
            self.sample_dir, output_dir, up_to_batch=3, shadow_check=False
        )
        self.assertEqual(result.recovery_batch_no, 2)
        self.assertEqual(
            result.report["snapshot"]["recovered_from_batch"], 2
        )
        manifest = SnapshotStore(
            os.path.join(output_dir, "snapshot")
        ).read_manifest()
        self.assertEqual(manifest.completed_batch_no, 3)
        self.assertEqual(set(manifest.stage_versions),
                         {"sampling", "scoring", "dedup"})

    def test_manifest_fingerprint_tampering_detected(self) -> None:
        output_dir = os.path.join(self.root, "tamper")
        run_workbench(
            self.sample_dir, output_dir, up_to_batch=1, shadow_check=False
        )
        manifest_path = os.path.join(output_dir, "snapshot", "manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["completed_batch_no"] = 99
        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, sort_keys=True))
        from sft_diag.errors import E_SNAPSHOT_CORRUPT
        result = run_workbench(
            self.sample_dir, output_dir, up_to_batch=1, shadow_check=False
        )
        self.assertTrue(
            any(f.code == E_SNAPSHOT_CORRUPT for f in result.failures)
        )


if __name__ == "__main__":
    unittest.main()
