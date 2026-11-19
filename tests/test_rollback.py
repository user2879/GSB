"""Tests for adaptive paths, atomic switch, recovery and manifest replacement."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_snap import (  # noqa: E402
    BlockSnapService,
    DomainError,
    E_PATH,
    JOB_COMMITTED,
    JOB_UNKNOWN,
    PATH_BITMAP,
    PATH_CHAIN,
    PATH_FULL,
    block_digest,
)
from block_snap.samples import PROFILES, build_profile  # noqa: E402


class RollbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="blocksnap-rb-")
        self.service = BlockSnapService(self.tmp)

    def tearDown(self) -> None:
        self.service.close()

    def test_path_selection_thresholds(self) -> None:
        profile = PROFILES["large_low_change"]
        build_profile(self.service, profile)
        base = self.service.list_snapshots(profile.name)[0]["snapshot_id"]
        sparse = self.service.plan_rollback(profile.name, base, bitmap_threshold=0.5, chain_span=99)
        self.assertEqual(sparse["path"], PATH_BITMAP)
        dense_profile = PROFILES["large_high_change"]
        build_profile(self.service, dense_profile)
        dbase = self.service.list_snapshots(dense_profile.name)[0]["snapshot_id"]
        dense = self.service.plan_rollback(dense_profile.name, dbase, bitmap_threshold=0.5, chain_span=99)
        self.assertEqual(dense["path"], PATH_FULL)
        chain = self.service.plan_rollback(dense_profile.name, dbase, chain_span=1)
        self.assertEqual(chain["path"], PATH_CHAIN)
        for plan in (sparse, dense, chain):
            self.assertIn("reason", plan)
            self.assertIn("threshold", plan)
            self.assertIn("diff_ratio", plan)

    def test_three_paths_byte_identical(self) -> None:
        for profile in PROFILES.values():
            with self.subTest(profile=profile.name):
                build_profile(self.service, profile)
                base = self.service.list_snapshots(profile.name)[0]["snapshot_id"]
                cross = self.service.cross_check_paths(profile.name, base)
                self.assertTrue(cross["identical"])
                expected = {
                    block_no: block_digest(
                        __import__("block_snap", fromlist=["deterministic_block"]).deterministic_block(
                            profile.seed, block_no, profile.block_size
                        )
                    )
                    for block_no in range(profile.block_count)
                }
                for mode in (PATH_BITMAP, PATH_FULL, PATH_CHAIN):
                    result = self.service.rollback(profile.name, base, force_path=mode)
                    self.assertEqual(result["state"], JOB_COMMITTED)
                    self.assertEqual(
                        self.service.open_volume(profile.name).block_digests(), expected
                    )

    def test_rollback_writes_through_staging_and_cleans_tmp(self) -> None:
        profile = PROFILES["small_many_snapshots"]
        build_profile(self.service, profile)
        base = self.service.list_snapshots(profile.name)[0]["snapshot_id"]
        self.service.rollback(profile.name, base)
        leftovers = [name for name in os.listdir(self.service.manager.tmp_dir) if name.endswith(".staging")]
        self.assertEqual(leftovers, [])
        jobs = self.service.manager.store.list_jobs(profile.name)
        self.assertTrue(all(job["state"] == JOB_COMMITTED for job in jobs))

    def test_crash_recovery_marks_unknown_and_reports_blocks(self) -> None:
        profile = PROFILES["large_high_change"]
        build_profile(self.service, profile)
        snaps = self.service.list_snapshots(profile.name)
        target = snaps[0]["snapshot_id"]
        with self.assertRaises(SystemExit):
            self.service.rollback(profile.name, target, force_path=PATH_FULL, crash_after_blocks=4)
        staging = [
            name for name in os.listdir(self.service.manager.tmp_dir) if name.endswith(".staging")
        ]
        self.assertEqual(len(staging), 1)
        marker_path = os.path.join(self.service.manager.tmp_dir, staging[0] + ".marker")
        with open(marker_path, encoding="utf-8") as handle:
            marker = json.load(handle)
        self.assertEqual(marker["verified"], 4)

        self.service.close()
        restarted = BlockSnapService(self.tmp)
        recovery = restarted.recover(profile.name)
        unknown = [r for r in recovery["reports"] if r["state"] == JOB_UNKNOWN]
        self.assertEqual(len(unknown), 1)
        report = unknown[0]
        self.assertEqual(report["verified_blocks"], 4)
        self.assertGreater(report["unverified_blocks"], 0)
        self.assertEqual(report["action"], "rerun_required")
        rerun = restarted.rollback(profile.name, target, verify_all_paths=True)
        self.assertEqual(rerun["state"], JOB_COMMITTED)
        token = restarted.completion_token(profile.name, target)
        self.assertTrue(token["committed"])
        self.assertEqual(token["snapshot_id"], target)
        restarted.close()
        self.service = BlockSnapService(self.tmp)

    def test_manifest_full_replacement_with_backup(self) -> None:
        profile = PROFILES["small_many_snapshots"]
        build_profile(self.service, profile)
        before = self.service.manifest_view(profile.name)
        entries = []
        for row in self.service.manager.store.list_snapshots(profile.name):
            refs = self.service.manager.store.ref_map(profile.name, row["snapshot_id"])
            entries.append({
                "snapshot_id": row["snapshot_id"],
                "name": row["name"],
                "version_no": row["version_no"],
                "parent_id": row["parent_id"],
                "bitmap_hex": bytes(row["bitmap"]).hex(),
                "refs": {str(b): h for b, h in refs.items()},
            })
        result = self.service.replace_manifest(profile.name, before["format_version"] + 5, entries)
        self.assertEqual(result["replaced_entries"], len(entries))
        self.assertTrue(result["version_changed"])
        self.assertTrue(os.path.exists(result["backup_path"]))
        self.assertEqual(os.stat(result["backup_path"]).st_mode & 0o200, 0)
        self.assertEqual(self.service.manifest_view(profile.name)["format_version"],
                         before["format_version"] + 5)
        # digest drift is detected when stored digest is tampered with
        self.service.manager.store.conn.execute(
            "UPDATE manifests SET content_digest=? WHERE volume_name=?", ("deadbeef", profile.name)
        )
        self.service.manager.store.commit()
        self.assertTrue(self.service.manifests.detect_drift(profile.name)["drift"])

    def test_bad_claimed_digest_rejected_without_mutation(self) -> None:
        profile = PROFILES["small_many_snapshots"]
        build_profile(self.service, profile)
        before = self.service.manifest_view(profile.name)
        entries = []
        for row in self.service.manager.store.list_snapshots(profile.name):
            refs = self.service.manager.store.ref_map(profile.name, row["snapshot_id"])
            entries.append({
                "snapshot_id": row["snapshot_id"], "name": row["name"],
                "version_no": row["version_no"], "parent_id": row["parent_id"],
                "bitmap_hex": bytes(row["bitmap"]).hex(),
                "refs": {str(b): h for b, h in refs.items()},
            })
        with self.assertRaises(DomainError):
            self.service.replace_manifest(profile.name, 77, entries, claimed_digest="nope")
        self.assertEqual(self.service.manifest_view(profile.name), before)

    def test_path_divergence_reports_first_block(self) -> None:
        profile = PROFILES["small_many_snapshots"]
        build_profile(self.service, profile)
        base = self.service.list_snapshots(profile.name)[0]["snapshot_id"]
        engine = self.service.rollback_engine
        good_target = engine.target_state(profile.name, base)
        divergent_target = dict(good_target)
        flipped = bytearray(divergent_target[5])
        flipped[0] ^= 0xAA
        divergent_target[5] = bytes(flipped)

        good_staging, good_marker = engine._build_staging(
            profile.name, base, good_target, PATH_BITMAP
        )
        bad_staging, bad_marker = engine._build_staging(
            profile.name, base, divergent_target, PATH_FULL
        )
        self.assertEqual(
            engine._first_divergent_block(good_staging, bad_staging, profile.name), 5
        )
        for path in (good_staging, bad_staging, good_marker, bad_marker):
            if os.path.exists(path):
                os.remove(path)

        original_builder = engine._build_staging

        def patched_builder(volume_name, snapshot_id, target, mode):
            if mode == PATH_FULL:
                target = dict(target)
                flipped = bytearray(target[5])
                flipped[0] ^= 0xAA
                target[5] = bytes(flipped)
            return original_builder(volume_name, snapshot_id, target, mode)

        engine._build_staging = patched_builder
        try:
            with self.assertRaises(DomainError) as caught:
                engine.cross_check_paths(profile.name, base)
        finally:
            engine._build_staging = original_builder
        self.assertEqual(caught.exception.code, E_PATH)
        self.assertEqual(caught.exception.block_no, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
