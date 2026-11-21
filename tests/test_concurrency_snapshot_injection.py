from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from candidate_exchange.engine import PipelineOptions, SimulatedTermination, run_pipeline
from candidate_exchange.injection import apply_injections

from tests.support import ensure_sample_data, normalized_state


class ConcurrencySnapshotInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = ensure_sample_data()

    def test_single_and_multi_thread_states_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            single = run_pipeline(
                PipelineOptions(
                    data_dir=self.data,
                    output_dir=root / "single",
                    snapshot_root=root / "snap_single",
                    workers=1,
                    snapshot_commit_interval=1,
                )
            )
            multi = run_pipeline(
                PipelineOptions(
                    data_dir=self.data,
                    output_dir=root / "multi",
                    snapshot_root=root / "snap_multi",
                    workers=4,
                    snapshot_commit_interval=2,
                )
            )
            self.assertEqual(normalized_state(single), normalized_state(multi))

    def test_snapshot_termination_ignores_partial_output_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            snapshots = root / "snapshots"
            with self.assertRaises(SimulatedTermination):
                run_pipeline(
                    PipelineOptions(
                        data_dir=self.data,
                        output_dir=output,
                        snapshot_root=snapshots,
                        workers=2,
                        snapshot_commit_interval=2,
                        crash_after_stage="detection",
                    )
                )
            self.assertFalse((output / "validation_report.json").exists())
            recovered = run_pipeline(
                PipelineOptions(
                    data_dir=self.data,
                    output_dir=output,
                    snapshot_root=snapshots,
                    workers=2,
                    snapshot_commit_interval=2,
                )
            )
            self.assertTrue(recovered["snapshot_recovery_at_start"]["recovered"])
            self.assertIn(
                "detection",
                recovered["snapshot_recovery_at_start"]["available_stages"],
            )
            self.assertTrue((output / "validation_report.json").exists())

    def test_all_injections_hit_and_failed_frames_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            injected = root / "injected"
            apply_injections(
                self.data,
                injected,
                self.data / "injections.json",
                force=True,
            )
            report = run_pipeline(
                PipelineOptions(
                    data_dir=injected,
                    output_dir=root / "out",
                    snapshot_root=root / "snapshots",
                    workers=4,
                    snapshot_commit_interval=2,
                )
            )
            self.assertTrue(report["injections"]["all_hit"])
            self.assertEqual(len(report["injections"]["hits"]), 7)
            failed = {
                (failure["series_name"], failure["frame_index"])
                for failure in report["failures"]
                if failure["frame_index"] is not None
            }
            self.assertFalse(
                any(
                    (track["series_name"], frame["frame_index"]) in failed
                    for track in report["tracks"]
                    for frame in track["frames"]
                )
            )


if __name__ == "__main__":
    unittest.main()
