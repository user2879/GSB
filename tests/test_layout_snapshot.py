import tempfile
import unittest
from pathlib import Path

from subtitle_workbench.ass_export import export_ass, verify_export
from subtitle_workbench.controller import WorkbenchController
from subtitle_workbench.errors import StaleVersionError
from subtitle_workbench.layout import AD_MIN_DURATION, overlaps, solve_layout
from subtitle_workbench.models import SubtitleItem
from subtitle_workbench.sample import generate_sample
from subtitle_workbench.snapshot import SnapshotStore


def make_item(item_id, start, end, *, is_ad=False, priority=2, text="测试"):
    return SubtitleItem(item_id, start, end, text,
                        "AUDIO_DESC" if is_ad else "ANNA",
                        "zh", priority, is_ad)


class LayoutSolverTests(unittest.TestCase):
    def setUp(self):
        self.project = generate_sample()
        self.result = solve_layout(self.project.items)

    def test_deterministic_byte_identical_solve(self):
        again = solve_layout(self.project.items)
        self.assertEqual(self.result.content_hash, again.content_hash)
        self.assertEqual(
            [cue.to_dict() for cue in self.result.cues],
            [cue.to_dict() for cue in again.cues])

    def test_no_same_lane_overlap_anywhere(self):
        cues = list(self.result.cues)
        for i, left in enumerate(cues):
            for right in cues[i + 1:]:
                if left.lane_index == right.lane_index:
                    self.assertFalse(overlaps(
                        left.item.start_frame, left.item.end_frame,
                        right.item.start_frame, right.item.end_frame))

    def test_every_cue_inside_declared_safe_area(self):
        config = self.result.config
        for cue in self.result.cues:
            rendered_bottom = cue.y + len(cue.wrap.lines) * config.lane_height
            self.assertGreaterEqual(cue.y, config.top_margin)
            self.assertLessEqual(rendered_bottom,
                                 config.height - config.bottom_margin)
            self.assertEqual(cue.x, config.side_margin)
            self.assertEqual(cue.width,
                             config.width - 2 * config.side_margin)
            self.assertLessEqual(cue.wrap.max_line_width,
                                 cue.width + 1e-9)

    def test_ad_uses_declared_top_band_and_longer_min_duration(self):
        ad_cues = [cue for cue in self.result.cues if cue.item.is_ad]
        for cue in self.result.cues:
            if cue.item.is_ad:
                self.assertGreaterEqual(cue.item.duration_frames(),
                                        AD_MIN_DURATION)
        # at least one AD sits in the declared top band; overflow is allowed
        self.assertTrue(any(cue.lane_index < self.result.config.ad_lanes
                            for cue in ad_cues))
        overflowed = [cue for cue in ad_cues
                      if cue.lane_index >= self.result.config.ad_lanes]
        for cue in overflowed:
            self.assertTrue(any(
                arb.item_id == cue.item.item_id
                for arb in self.result.arbitrations))

    def test_ad_yield_rule_records_arbitration_and_never_drops(self):
        ids = {cue.item.item_id for cue in self.result.cues}
        self.assertEqual(ids, {item.item_id for item in self.project.items})
        for arb in self.result.arbitrations:
            self.assertIn(arb.reason,
                          {"ad_yields_to_dialogue",
                           "declared_band_full_overflow"})
            self.assertGreaterEqual(arb.assigned_lane,
                                    self.result.config.ad_lanes)

    def test_explicit_ad_yields_to_dialogue_conflict(self):
        items = (
            make_item("AD1", 20, 80, is_ad=True, priority=2),
            make_item("AD2", 20, 80, is_ad=True, priority=2),
            make_item("AD3", 20, 80, is_ad=True, priority=3),
            make_item("D1", 20, 80, is_ad=False, priority=1),
            make_item("D2", 20, 80, is_ad=False, priority=2),
            make_item("D3", 20, 80, is_ad=False, priority=3),
            make_item("D4", 20, 80, is_ad=False, priority=4),
        )
        result = solve_layout(items)
        self.assertEqual(len(result.cues), 7)
        self.assertTrue(result.arbitrations)
        # all cues survive; an AD (never the priority-1 dialogue) is moved
        for arb in result.arbitrations:
            self.assertTrue(arb.item_id.startswith("AD"))

    def test_ordering_total_order_tie_broken_by_id(self):
        items = (make_item("B", 0, 50, priority=1),
                 make_item("A", 0, 50, priority=1))
        result = solve_layout(items)
        lane = {cue.item.item_id: cue.lane_index for cue in result.cues}
        self.assertLess(lane["A"], lane["B"])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self.tmp.name))
        project = generate_sample()
        self.store.commit(project.items, solve_layout(project.items))
        self.controller = WorkbenchController(self.store)
        self.first_id = project.items[0].item_id

    def tearDown(self):
        self.tmp.cleanup()

    def _expect_code(self, item_id, lane, start, end, code):
        self.controller.begin_drag(item_id)
        preview = self.controller.preview_drag(item_id, lane, start, end)
        self.assertIsNotNone(preview.error)
        self.assertEqual(preview.error.code, code)
        self.controller.cancel_drag()
        return preview.error

    def test_legal_preview_then_commit_creates_new_snapshot(self):
        self.controller.begin_drag(self.first_id)
        preview = self.controller.preview_drag(self.first_id, 5, 60, 150)
        self.assertTrue(preview.accepted)
        _, snapshot = self.controller.commit_drag()
        self.assertEqual(snapshot.version, 2)
        moved = next(item for item in snapshot.items
                     if item.item_id == self.first_id)
        self.assertEqual((moved.start_frame, moved.end_frame), (60, 150))

    def test_rejected_drag_codes_and_state_untouched(self):
        before = self.controller.committed_items
        err = self._expect_code(self.first_id, 999, 10, 20,
                                "E_DRAG_BOUNDS")
        self.assertEqual(err.conflict_item_id, self.first_id)
        self._expect_code(self.first_id, 2, 300, 300, "E_TIME_INVALID")
        current = self.store.load_current()
        lanes = {cue.item.item_id: cue.lane_index
                 for cue in current.layout.cues}
        high = next(item for item in current.items if item.priority == 1)
        victim = next(item for item in current.items
                      if item.item_id != high.item_id
                      and item.priority > high.priority)
        err = self._expect_code(victim.item_id, lanes[high.item_id],
                                high.start_frame + 1,
                                high.end_frame - 1,
                                "E_TRACK_CONFLICT")
        self.assertEqual(err.conflict_item_id, high.item_id)
        self.assertEqual(self.controller.committed_items, before)
        self.assertEqual(self.store.load_current().version, 1)

    def test_cancel_restores_exact_state(self):
        before = self.controller.committed_items
        self.controller.begin_drag(self.first_id)
        self.controller.preview_drag(self.first_id, 3, 900, 990)
        self.controller.cancel_drag()
        self.assertEqual(self.controller.committed_items, before)

    def test_undo_restores_immutable_snapshot(self):
        self.controller.begin_drag(self.first_id)
        self.controller.preview_drag(self.first_id, 5, 60, 150)
        self.controller.commit_drag()
        self.assertEqual(self.store.load_current().version, 2)
        self.controller.undo(1)
        self.assertEqual(self.store.load_current().version, 1)
        with self.assertRaises(StaleVersionError):
            self.controller.undo(99)

    def test_stale_commit_base_version_rejected(self):
        with self.assertRaises(StaleVersionError):
            self.store.commit(
                self.controller.committed_items,
                solve_layout(self.controller.committed_items),
                expected_previous_version=42)


class ExportVerificationTests(unittest.TestCase):
    def test_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SnapshotStore(root)
            project = generate_sample()
            snapshot = store.commit(project.items,
                                    solve_layout(project.items))
            ass_path = root / "out.ass"
            export_ass(snapshot, ass_path)
            report = verify_export(ass_path,
                                   store._snapshot_path(snapshot.version))
            self.assertTrue(report.ok)
            self.assertEqual(report.details["matched_cues"],
                             len(project.items))

            rows = ass_path.read_text(encoding="utf-8").splitlines()
            index = next(i for i, row in enumerate(rows)
                         if row.startswith("Dialogue:") and "SUB-005" in row)
            parts = rows[index].split(",", 9)
            parts[2] = "0:00:59.99"
            rows[index] = ",".join(parts)
            tampered = root / "bad.ass"
            tampered.write_text("\n".join(rows), encoding="utf-8")
            bad = verify_export(tampered,
                                store._snapshot_path(snapshot.version))
            self.assertFalse(bad.ok)
            self.assertIn("SUB-005", bad.first_mismatch)
            self.assertEqual(bad.details["field"], "event_end_frame")

    def test_count_alone_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SnapshotStore(root)
            project = generate_sample()
            snapshot = store.commit(project.items,
                                    solve_layout(project.items))
            ass_path = root / "out.ass"
            export_ass(snapshot, ass_path)
            rows = ass_path.read_text(encoding="utf-8").splitlines()
            # swap lane indices of two cues but keep every cue present
            dialogue_rows = [i for i, row in enumerate(rows)
                             if row.startswith("Dialogue:")]
            first, second = dialogue_rows[0], dialogue_rows[1]
            p1 = rows[first].split(",", 9)
            p2 = rows[second].split(",", 9)
            p1[0], p2[0] = p2[0], p1[0]
            rows[first], rows[second] = ",".join(p1), ",".join(p2)
            swapped = root / "swapped.ass"
            swapped.write_text("\n".join(rows), encoding="utf-8")
            report = verify_export(swapped,
                                   store._snapshot_path(snapshot.version))
            self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
