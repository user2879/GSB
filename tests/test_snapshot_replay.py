from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from pagecache.aggregate import FixedPointAggregator
from pagecache.cache import PageCache
from pagecache.errors import EngineError, ErrorCode
from pagecache.manifest import read_manifest
from pagecache.replay import replay_from_snapshot
from pagecache.snapshot import load_snapshot, restore_snapshot, save_snapshot
from tests.helpers import build_snapshot_run, make_samples


class SnapshotReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data, cls.meta = make_samples(cls.root)
        cls.run_dir = cls.root / "snapshot-run"
        cls.artifacts = build_snapshot_run(
            data=cls.data,
            meta=cls.meta,
            run_dir=cls.run_dir,
            position=5,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def _new_engine(self, overflow_dir: Path) -> tuple[PageCache, FixedPointAggregator]:
        artifacts = self.artifacts
        cache = PageCache(
            manifest=read_manifest(artifacts["manifest_path"]),
            pages_dir=artifacts["pages_dir"],
            overflow_dir=overflow_dir,
            budget_bytes=int(artifacts["budget_bytes"]),
            page_size=int(artifacts["page_size"]),
        )
        aggregator = FixedPointAggregator(str(self.meta["group_column"]), str(self.meta["value_column"]))
        return cache, aggregator

    def test_restore_is_wholesale_and_lists_lost_and_new_segments(self) -> None:
        modified = self.root / "modified-overflow"
        shutil.copytree(self.artifacts["overflow_dir"], modified)
        snapshot = load_snapshot(self.artifacts["snapshot_path"])
        removed_ref = str(snapshot["cache"]["overflow_pages"][0]["overflow_ref"])
        (modified / removed_ref).unlink()
        source_segment = sorted(modified.glob("*.seg"), key=lambda item: item.name)[0]
        shutil.copy2(source_segment, modified / "zz-new.segment.seg")

        cache, aggregator = self._new_engine(modified)
        cache.get_page("p0008")
        before = cache.state()
        self.assertNotEqual(before, snapshot["cache"])
        report = restore_snapshot(snapshot, cache=cache, aggregator=aggregator, exact_disk=False)
        self.assertEqual(report["replacement_mode"], "wholesale")
        self.assertEqual(report["disk_diff"]["lost_overflow_segments"], [removed_ref])
        self.assertEqual(report["disk_diff"]["new_overflow_segments"], ["zz-new.segment.seg"])
        restored_state = cache.state()
        self.assertEqual(
            [row["page_id"] for row in restored_state["resident_pages"]],
            [row["page_id"] for row in snapshot["cache"]["resident_pages"]],
        )
        self.assertEqual(
            [row["page_id"] for row in restored_state["overflow_pages"]],
            [row["page_id"] for row in snapshot["cache"]["overflow_pages"]],
        )
        self.assertEqual(aggregator.state(), snapshot["aggregation"])

    def test_replay_succeeds_on_unchanged_workspace(self) -> None:
        cache, aggregator = self._new_engine(self.artifacts["overflow_dir"])
        result = replay_from_snapshot(self.artifacts["snapshot_path"], cache=cache, aggregator=aggregator)
        self.assertEqual(result["status"], "reproducible")
        self.assertEqual(result["compared_records"], 12 - 5)
        self.assertEqual(result["final_output_sha256"], self.artifacts["snapshot"]["reference_final_output_sha256"])

    def test_missing_overflow_segment_is_e_replay_with_basis(self) -> None:
        broken = self.root / "missing-segment-overflow"
        shutil.copytree(self.artifacts["overflow_dir"], broken)
        snapshot = load_snapshot(self.artifacts["snapshot_path"])
        removed_ref = str(snapshot["cache"]["overflow_pages"][0]["overflow_ref"])
        expected_page = str(snapshot["cache"]["overflow_pages"][0]["page_id"])
        (broken / removed_ref).unlink()
        cache, aggregator = self._new_engine(broken)
        with self.assertRaises(EngineError) as caught:
            replay_from_snapshot(self.artifacts["snapshot_path"], cache=cache, aggregator=aggregator)
        self.assertIs(caught.exception.code, ErrorCode.E_REPLAY)
        self.assertEqual(caught.exception.page_id, expected_page)
        self.assertEqual(caught.exception.details["basis"], "overflow_segment_missing_or_changed")

    def test_first_trace_mismatch_locates_page_and_field(self) -> None:
        modified_snapshot = self.root / "mismatch-snapshot.json"
        snapshot = load_snapshot(self.artifacts["snapshot_path"])
        future_index = int(snapshot["position"]) + 1
        snapshot["reference_trace"][future_index]["swap_in_count"] += 1000
        save_snapshot(modified_snapshot, snapshot)
        cache, aggregator = self._new_engine(self.artifacts["overflow_dir"])
        with self.assertRaises(EngineError) as caught:
            replay_from_snapshot(modified_snapshot, cache=cache, aggregator=aggregator)
        error = caught.exception
        self.assertIs(error.code, ErrorCode.E_REPLAY)
        self.assertEqual(error.details["field"], "swap_in_count")
        self.assertEqual(error.details["access_index"], future_index)
        self.assertEqual(error.page_id, snapshot["scan_order"][future_index])

    def test_manifest_version_change_is_e_replay(self) -> None:
        modified_snapshot = self.root / "version-changed-snapshot.json"
        snapshot = load_snapshot(self.artifacts["snapshot_path"])
        snapshot["manifest_format_versions"] = [2]
        save_snapshot(modified_snapshot, snapshot)
        cache, aggregator = self._new_engine(self.artifacts["overflow_dir"])
        with self.assertRaises(EngineError) as caught:
            replay_from_snapshot(modified_snapshot, cache=cache, aggregator=aggregator)
        self.assertIs(caught.exception.code, ErrorCode.E_REPLAY)
        self.assertEqual(caught.exception.details["basis"], "manifest_format_version_changed")

    def test_fixed_point_version_change_is_e_replay(self) -> None:
        modified_snapshot = self.root / "fixed-version-changed-snapshot.json"
        snapshot = load_snapshot(self.artifacts["snapshot_path"])
        snapshot["fixed_point"]["version"] = "fixed-v0-other"
        save_snapshot(modified_snapshot, snapshot)
        cache, aggregator = self._new_engine(self.artifacts["overflow_dir"])
        with self.assertRaises(EngineError) as caught:
            replay_from_snapshot(modified_snapshot, cache=cache, aggregator=aggregator)
        self.assertIs(caught.exception.code, ErrorCode.E_REPLAY)
        self.assertEqual(caught.exception.details["basis"], "fixed_point_version_changed")


if __name__ == "__main__":
    unittest.main()
