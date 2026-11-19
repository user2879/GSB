from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mocap_workbench.constants import DEFAULT_SEED
from mocap_workbench.detection import (
    compute_evidence,
    detect,
    detection_payload,
    thresholds_from_mapping,
)
from mocap_workbench.errors import BusinessError, ErrorCode
from mocap_workbench.evaluation import evaluate_full
from mocap_workbench.generator import generate_bundle, load_motion
from mocap_workbench.io import canonical_json_bytes
from mocap_workbench.model import Thresholds
from mocap_workbench.persistence import StateStore, simulate_incomplete_write
from mocap_workbench.pipeline import (
    apply_incremental,
    build_with_evidence,
    check_consistency,
    rebuild_full_for_consistency,
)
from mocap_workbench.preview import PreviewEngine, SnapshotManager
from mocap_workbench.repair import repair_payload


class WorkbenchCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="mocap-tests-")
        cls.root = Path(cls.temp.name)
        cls.bundle = generate_bundle(cls.root / "a", 1200, DEFAULT_SEED)
        cls.evidence = compute_evidence(cls.bundle.record)
        cls.thresholds = thresholds_from_mapping()
        cls.detection = detect(
            cls.bundle.record, cls.bundle.defects, cls.thresholds, cls.evidence
        )
        cls.baseline = build_with_evidence(
            cls.bundle.record,
            cls.bundle.truth,
            cls.bundle.defects,
            cls.evidence,
            cls.thresholds,
            frozenset(),
            0,
            committed=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_generation_is_byte_identical(self) -> None:
        second = generate_bundle(self.root / "b", 1200, DEFAULT_SEED)
        for name in ("record.json", "truth.json", "defects.json"):
            self.assertEqual(
                (self.bundle.record_path.parent / name).read_bytes(),
                (second.record_path.parent / name).read_bytes(),
                name,
            )

    def test_detection_is_exact_and_repeatable(self) -> None:
        second_evidence = compute_evidence(self.bundle.record)
        second_detection = detect(
            self.bundle.record,
            self.bundle.defects,
            thresholds_from_mapping(),
            second_evidence,
        )
        first_bytes = canonical_json_bytes(
            detection_payload(self.detection, self.evidence, self.bundle.record)
        )
        second_bytes = canonical_json_bytes(
            detection_payload(second_detection, second_evidence, self.bundle.record)
        )
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(self.detection.comparison["exact_match"])
        self.assertEqual(self.detection.comparison["missed_count"], 0)
        self.assertEqual(self.detection.comparison["false_positive_count"], 0)

    def test_clean_frames_are_byte_preserved(self) -> None:
        for frame in range(self.bundle.record.frame_count):
            for joint in range(self.bundle.record.joint_count):
                if not self.detection.joint_bad[frame, joint]:
                    np.testing.assert_array_equal(
                        self.baseline.repair.quats[frame, joint],
                        self.bundle.record.quats[frame, joint],
                    )
            if not self.detection.root_bad[frame]:
                np.testing.assert_array_equal(
                    self.baseline.repair.roots[frame],
                    self.bundle.record.roots[frame],
                )
        self.assertTrue(self.baseline.evaluation.trusted)

    def test_repair_is_byte_identical_for_same_inputs(self) -> None:
        second = build_with_evidence(
            self.bundle.record,
            self.bundle.truth,
            self.bundle.defects,
            self.evidence,
            self.thresholds,
            frozenset(),
            0,
            committed=True,
        )
        first_bytes = canonical_json_bytes(
            repair_payload(self.baseline.to_meta_dict(), self.baseline.record, self.baseline.repair)
        )
        second_bytes = canonical_json_bytes(
            repair_payload(second.to_meta_dict(), second.record, second.repair)
        )
        self.assertEqual(first_bytes, second_bytes)

    def test_threshold_increment_matches_full_recompute(self) -> None:
        changed = replace(self.thresholds, joint_jump_deg=5.0)
        incremental = apply_incremental(
            self.baseline, self.bundle.defects, thresholds=changed, sequence=1
        )
        self.assertTrue(incremental.closure.dirty_joint_coords)
        self.assertEqual(
            set(incremental.closure.dirty_joint_coords),
            set(incremental.closure.actual_recomputed_joint_coords),
        )
        full = rebuild_full_for_consistency(incremental, self.bundle.defects)
        report = check_consistency(incremental, full)
        self.assertTrue(report.consistent)
        self.assertEqual(report.mismatch_count, 0)
        self.assertIsNone(report.first_mismatch)
        dirty_frames = {frame for frame, _ in incremental.closure.dirty_joint_coords}
        for frame in range(self.baseline.frame_count):
            if frame not in dirty_frames:
                self.assertEqual(
                    self.baseline.repair.frame_hashes[frame],
                    incremental.repair.frame_hashes[frame],
                )

    def test_combined_threshold_and_keyframe_closure_matches_full(self) -> None:
        active_auto: set[int] = set(self.baseline.anchors.root_auto)
        for anchors in self.baseline.anchors.joint_auto:
            active_auto.update(anchors)
        old_anchor = next(frame for frame in sorted(active_auto) if frame not in (0, self.baseline.frame_count - 1))
        new_anchor = old_anchor + 3
        while (
            self.baseline.detection.joint_bad[new_anchor, :].any()
            or self.baseline.detection.root_bad[new_anchor]
            or new_anchor in active_auto
        ):
            new_anchor += 1
        incremental = apply_incremental(
            self.baseline,
            self.bundle.defects,
            thresholds=replace(self.thresholds, joint_jump_deg=5.0),
            keyframes=frozenset(self.baseline.keyframes | {new_anchor}),
            suppressed_auto=frozenset(self.baseline.suppressed_auto | {old_anchor}),
            sequence=1,
        )
        self.assertTrue(any(key.startswith("threshold:") for key in incremental.closure.changed_keys))
        self.assertTrue(
            any(key.startswith("auto:") or key.startswith("keyframe:") for key in incremental.closure.changed_keys)
        )
        full = rebuild_full_for_consistency(incremental, self.bundle.defects)
        report = check_consistency(incremental, full)
        self.assertTrue(report.consistent)
        self.assertEqual(report.mismatch_count, 0)

    def test_editable_auto_anchor_closes_and_updates_only_closure(self) -> None:
        old_anchor = None
        active_auto: set[int] = set(self.baseline.anchors.root_auto)
        for anchors in self.baseline.anchors.joint_auto:
            active_auto.update(anchors)
        for candidate in sorted(active_auto):
            if candidate not in (0, self.baseline.frame_count - 1):
                old_anchor = candidate
                break
        self.assertIsNotNone(old_anchor)
        new_anchor = old_anchor + 4
        while new_anchor < self.baseline.frame_count and (
            self.baseline.detection.joint_bad[new_anchor, :].any()
            or self.baseline.detection.root_bad[new_anchor]
            or new_anchor in active_auto
        ):
            new_anchor += 1
        incremental = apply_incremental(
            self.baseline,
            self.bundle.defects,
            keyframes=frozenset(self.baseline.keyframes | {new_anchor}),
            suppressed_auto=frozenset(self.baseline.suppressed_auto | {old_anchor}),
            sequence=1,
        )
        self.assertTrue(incremental.closure.dirty_joint_coords or incremental.closure.dirty_root_frames)
        self.assertEqual(
            set(incremental.closure.dirty_joint_coords),
            set(incremental.closure.actual_recomputed_joint_coords),
        )
        full = rebuild_full_for_consistency(incremental, self.bundle.defects)
        self.assertTrue(check_consistency(incremental, full).consistent)
        dirty_frames = {frame for frame, _ in incremental.closure.dirty_joint_coords}
        for frame in range(self.baseline.frame_count):
            if frame not in dirty_frames:
                self.assertEqual(
                    self.baseline.repair.frame_hashes[frame],
                    incremental.repair.frame_hashes[frame],
                )

    def test_evaluation_marks_untrusted_details(self) -> None:
        entry = next(entry for entry in self.detection.entries if entry.joint != "Root")
        joint = self.bundle.record.joint_names.index(entry.joint)
        quats = self.baseline.repair.quats.copy()
        quats[entry.frame, joint, :] = [1.0, 0.0, 0.0, 0.0]
        damaged = replace(self.baseline.repair, quats=quats)
        evaluation = evaluate_full(
            self.bundle.record, self.bundle.truth, self.detection, damaged
        )
        self.assertFalse(evaluation.trusted)
        self.assertEqual(evaluation.summary["publication_status"], "WITHHELD_UNTRUSTED")
        matching = [
            detail
            for detail in evaluation.details
            if detail.frame == entry.frame and detail.joint == entry.joint
        ]
        self.assertTrue(matching)
        self.assertFalse(matching[0].trusted)

    def test_business_error_codes(self) -> None:
        with self.assertRaises(BusinessError) as invalid_threshold:
            thresholds_from_mapping({"joint_jump_deg": 0.0})
        self.assertEqual(invalid_threshold.exception.code, ErrorCode.E_THRESHOLD_RANGE)
        with self.assertRaises(BusinessError) as missing_frame:
            self.bundle.record.check_frame(999999)
        self.assertEqual(missing_frame.exception.code, ErrorCode.E_FRAME_MISSING)
        with self.assertRaises(BusinessError) as missing_file:
            load_motion(self.root / "does-not-exist.json")
        self.assertEqual(missing_file.exception.code, ErrorCode.E_FRAME_MISSING)

        malformed_dir = self.root / "malformed"
        malformed_dir.mkdir(exist_ok=True)
        payload = json.loads(self.bundle.record_path.read_text(encoding="utf-8"))
        payload["frames"][0]["joints"]["Hips"] = [2.0, 0.0, 0.0, 0.0]
        malformed_path = malformed_dir / "record.json"
        malformed_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(BusinessError) as invalid_quaternion:
            load_motion(malformed_path)
        self.assertEqual(invalid_quaternion.exception.code, ErrorCode.E_QUAT_INVALID)

    def test_budget_overrun_reports_business_error(self) -> None:
        from mocap_workbench import preview as preview_module

        old_budget = preview_module.PREVIEW_BUDGET_MS[1200]
        try:
            preview_module.PREVIEW_BUDGET_MS[1200] = 0.0
            result = PreviewEngine().generate(self.baseline, self.bundle.defects)
            self.assertTrue(result.over_budget)
            self.assertEqual(result.budget_error["code"], ErrorCode.E_BUDGET_EXCEEDED.value)
            with self.assertRaises(BusinessError) as error:
                result.raise_if_over_budget()
            self.assertEqual(error.exception.code, ErrorCode.E_BUDGET_EXCEEDED)
        finally:
            preview_module.PREVIEW_BUDGET_MS[1200] = old_budget

    def test_stale_background_snapshot_is_rejected(self) -> None:
        manager = SnapshotManager(self.baseline)
        manager.submit_preview(self.bundle.defects, keyframes=frozenset({10}))
        changed = apply_incremental(
            self.baseline,
            self.bundle.defects,
            thresholds=replace(self.thresholds, joint_jump_deg=5.0),
            sequence=1,
            committed=True,
        )
        manager.swap_committed(changed)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            manager.poll()
            if manager.last_error:
                break
            time.sleep(0.02)
        self.assertIsNotNone(manager.last_error)
        self.assertEqual(manager.last_error["code"], ErrorCode.E_SNAPSHOT_STALE.value)
        manager.shutdown()

    def test_stale_snapshot_rejected_when_only_version_label_changes(self) -> None:
        manager = SnapshotManager(self.baseline)
        manager.submit_preview(self.bundle.defects, keyframes=frozenset({10}))
        relabeled = replace(
            self.baseline,
            version=replace(self.baseline.version, sequence=self.baseline.version.sequence + 1),
        )
        self.assertEqual(relabeled.content_hash, self.baseline.content_hash)
        self.assertNotEqual(relabeled.version.label, self.baseline.version.label)
        manager.swap_committed(relabeled)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            manager.poll()
            if manager.last_error:
                break
            time.sleep(0.02)
        self.assertIsNotNone(manager.last_error)
        self.assertEqual(manager.last_error["code"], ErrorCode.E_SNAPSHOT_STALE.value)
        self.assertEqual(
            manager.last_error["context"]["expected_content_hash"],
            manager.last_error["context"]["current_content_hash"],
        )
        self.assertNotEqual(
            manager.last_error["context"]["expected_version_label"],
            manager.last_error["context"]["current_version_label"],
        )
        manager.shutdown()

    def test_commit_rejects_same_content_hash_with_changed_version_label(self) -> None:
        manager = SnapshotManager(self.baseline)
        manager.submit_preview(self.bundle.defects, keyframes=frozenset({10}))
        deadline = time.time() + 5.0
        while time.time() < deadline and manager.preview is None:
            manager.poll()
            time.sleep(0.02)
        self.assertIsNotNone(manager.preview)
        relabeled = replace(
            self.baseline,
            version=replace(self.baseline.version, sequence=self.baseline.version.sequence + 1),
        )
        self.assertEqual(relabeled.content_hash, self.baseline.content_hash)
        with manager._lock:
            manager._current = relabeled
        with self.assertRaises(BusinessError) as stale:
            manager.commit_preview(StateStore(self.root / "unused-store"), self.bundle.defects)
        self.assertEqual(stale.exception.code, ErrorCode.E_SNAPSHOT_STALE)
        self.assertEqual(
            stale.exception.context["base_content_hash"],
            stale.exception.context["current_content_hash"],
        )
        self.assertNotEqual(
            stale.exception.context["base_version_label"],
            stale.exception.context["current_version_label"],
        )
        manager.shutdown()

    def test_consistency_reports_first_mismatch_in_frame_joint_order(self) -> None:
        full = rebuild_full_for_consistency(self.baseline, self.bundle.defects)
        quats = full.repair.quats.copy()
        frame = 10
        for joint_name in ("Spine", "Chest"):
            joint = full.record.joint_names.index(joint_name)
            quats[frame, joint, 1] += 0.5
        corrupted_full = replace(full, repair=replace(full.repair, quats=quats))
        report = check_consistency(self.baseline, corrupted_full)
        self.assertFalse(report.consistent)
        self.assertEqual(report.mismatch_count, 2)
        self.assertIsNotNone(report.first_mismatch)
        self.assertEqual(report.first_mismatch["frame"], frame)
        self.assertEqual(report.first_mismatch["joint"], "Chest")


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="mocap-persist-tests-")
        cls.root = Path(cls.temp.name)
        cls.bundle = generate_bundle(cls.root / "data", 1200, DEFAULT_SEED)
        cls.evidence = compute_evidence(cls.bundle.record)
        cls.snapshot = build_with_evidence(
            cls.bundle.record,
            cls.bundle.truth,
            cls.bundle.defects,
            cls.evidence,
            thresholds_from_mapping(),
            frozenset(),
            0,
            committed=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_clean_restart_commit_pointer_and_orphan_recovery(self) -> None:
        store = StateStore(self.root, "persist-test")
        store.begin_session()
        info = store.commit(self.snapshot, self.bundle.defects)
        store.close_cleanly()

        clean_store = StateStore(self.root, "persist-test")
        clean_recovery = clean_store.begin_session()
        self.assertTrue(clean_recovery.previous_exit_clean)
        self.assertEqual(clean_store.load_current().content_hash, self.snapshot.content_hash)

        simulate_incomplete_write(clean_store)
        crash_store = StateStore(self.root, "persist-test")
        crash_recovery = crash_store.begin_session()
        self.assertTrue(crash_recovery.previous_session_present)
        self.assertFalse(crash_recovery.previous_exit_clean)
        self.assertGreaterEqual(crash_recovery.orphan_staging_files_removed, 1)
        self.assertGreaterEqual(crash_recovery.orphan_blobs_removed, 1)
        restored = crash_store.restore_committed_snapshot(
            self.bundle.record, self.bundle.truth, self.bundle.defects
        )
        self.assertIsNotNone(restored)
        self.assertEqual(restored.content_hash, self.snapshot.content_hash)
        self.assertEqual(restored.repair.repair_hash, self.snapshot.repair.repair_hash)
        crash_store.close_cleanly()


if __name__ == "__main__":
    unittest.main()
