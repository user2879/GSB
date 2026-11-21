from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from candidate_exchange.edits import apply_operation, copy_dataset
from candidate_exchange.engine import PipelineOptions, run_pipeline
from candidate_exchange.util import read_json

from tests.support import ensure_sample_data, normalized_state


class IncrementalTests(unittest.TestCase):
    def test_incremental_final_state_matches_blank_full_rebuild(self) -> None:
        data = ensure_sample_data()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            editable = root / "editable"
            copy_dataset(data, editable, force=True)
            edit_snapshot = root / "snapshots_edit"
            run_pipeline(
                PipelineOptions(
                    data_dir=editable,
                    output_dir=root / "baseline",
                    snapshot_root=edit_snapshot,
                    workers=4,
                    snapshot_commit_interval=2,
                )
            )
            operations = read_json(editable / "edit_sequence.json")["operations"]
            incremental = None
            for index, operation in enumerate(operations):
                apply_operation(editable, operation)
                incremental = run_pipeline(
                    PipelineOptions(
                        data_dir=editable,
                        output_dir=root / f"step_{index}",
                        snapshot_root=edit_snapshot,
                        workers=4,
                        snapshot_commit_interval=2,
                    )
                )
            full = run_pipeline(
                PipelineOptions(
                    data_dir=editable,
                    output_dir=root / "full",
                    snapshot_root=root / "snapshots_full",
                    workers=1,
                    snapshot_commit_interval=1,
                    force_full=True,
                )
            )
            self.assertIsNotNone(incremental)
            self.assertEqual(normalized_state(full), normalized_state(incremental))
            self.assertGreater(incremental["summary"]["reused_frame_count"], 0)
            self.assertGreater(incremental["summary"]["reused_track_count"], 0)
            self.assertGreater(incremental["summary"]["recomputed_track_count"], 0)


if __name__ == "__main__":
    unittest.main()
