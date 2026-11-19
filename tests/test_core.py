"""Unit tests for validation, CoW snapshots and the manifest/audit rules."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_snap import (  # noqa: E402
    Bitmap,
    BlockSnapService,
    DomainError,
    E_CHAIN,
    E_INPUT,
    E_INTEGRITY,
    E_SHAPE,
    block_digest,
    deterministic_block,
    parse_change_spec,
)
from block_snap.samples import PROFILES, build_profile, reset_workspace  # noqa: E402


class ServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="blocksnap-test-")
        self.service = BlockSnapService(self.tmp)
        self.profile = PROFILES["small_many_snapshots"]
        build_profile(self.service, self.profile)
        self.volume_name = self.profile.name

    def tearDown(self) -> None:
        self.service.close()

    def test_deterministic_content_is_reproducible(self) -> None:
        first = deterministic_block("seed", 3, 512, 7)
        second = deterministic_block("seed", 3, 512, 7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, deterministic_block("seed", 3, 512, 8))

    def test_change_spec_parsing_rejects_bad_input(self) -> None:
        self.assertEqual(parse_change_spec("0a:ff,1c:03"), [(10, 255), (28, 3)])
        for bad in ("", "nope", "0a", "zz:01", "0a:100", "0a:01,0a:02"):
            with self.subTest(bad=bad):
                with self.assertRaises(DomainError) as caught:
                    parse_change_spec(bad)
                self.assertEqual(caught.exception.code, E_INPUT)
                self.assertIsNotNone(caught.exception.field)

    def test_unsupported_block_size_rejected(self) -> None:
        with self.assertRaises(DomainError) as caught:
            self.service.create_volume("bad-size", 1000, 4)
        self.assertEqual(caught.exception.code, E_INPUT)
        self.assertEqual(caught.exception.field, "block_size")

    def test_out_of_range_block_no_and_shape(self) -> None:
        volume = self.service.open_volume(self.volume_name)
        with self.assertRaises(DomainError) as caught:
            volume.read_block(999)
        self.assertEqual(caught.exception.code, E_INPUT)
        with open(volume.path, "ab") as handle:
            handle.write(b"extra")
        with self.assertRaises(DomainError) as caught:
            self.service.verify_volume_shape(self.volume_name)
        self.assertEqual(caught.exception.code, E_SHAPE)

    def test_rejected_change_spec_writes_nothing(self) -> None:
        volume = self.service.open_volume(self.volume_name)
        before = volume.whole_digest()
        with self.assertRaises(DomainError):
            volume.inject_changes("00:11,bad-token")
        self.assertEqual(volume.whole_digest(), before)

    def test_duplicate_snapshot_name_rejected(self) -> None:
        with self.assertRaises(DomainError) as caught:
            self.service.create_snapshot(self.volume_name, self.profile.snapshot_names[0])
        self.assertEqual(caught.exception.code, E_INPUT)

    def test_snapshot_versions_monotonic_and_bitmap_compatible(self) -> None:
        snaps = self.service.list_snapshots(self.volume_name)
        self.assertEqual([s["version_no"] for s in snaps], [1, 2, 3, 4, 5])
        self.assertIsNone(snaps[0]["parent_id"])
        for earlier, later in zip(snaps, snaps[1:]):
            self.assertEqual(later["parent_id"], earlier["snapshot_id"])
        row = self.service.manager.store.list_snapshots(self.volume_name)[1]
        bitmap = Bitmap(self.profile.block_count, row["bitmap"])
        self.assertEqual(len(bitmap.to_bytes()), (self.profile.block_count + 7) // 8)

    def test_snapshot_does_not_copy_whole_volume(self) -> None:
        # only changed blocks (plus the base snapshot's initial blocks) live
        # in the content store; there is no per-snapshot full image
        block_files = os.listdir(self.service.manager.blocks_dir)
        stored = sum(
            len(self.service.manager.store.ref_map(self.volume_name, s["snapshot_id"]))
            for s in self.service.list_snapshots(self.volume_name)
        )
        self.assertLess(stored, self.profile.block_count * len(self.profile.snapshot_names))
        self.assertTrue(all(len(name) == 64 for name in block_files))

    def test_manifest_digest_stable_and_recomputed(self) -> None:
        view = self.service.manifest_view(self.volume_name)
        self.assertTrue(view["digest_match"])
        again = self.service.manifest_view(self.volume_name)
        self.assertEqual(view["recomputed_digest"], again["recomputed_digest"])

    def test_integrity_error_on_corrupt_block(self) -> None:
        snaps = self.service.list_snapshots(self.volume_name)
        target = snaps[1]["snapshot_id"]
        refs = self.service.manager.store.ref_map(self.volume_name, target)
        block_no, block_hash = next(iter(refs.items()))
        object_path = self.service.manager.blocks.path_for(block_hash)
        with open(object_path, "r+b") as handle:
            handle.seek(0)
            handle.write(b"\xff" * 16)
        with self.assertRaises(DomainError) as caught:
            self.service.manager.verify_block_integrity(self.volume_name, target)
        self.assertEqual(caught.exception.code, E_INTEGRITY)
        self.assertEqual(caught.exception.block_no, block_no)
        logged = self.service.query_logs(code=E_INTEGRITY)
        self.assertTrue(logged and logged[-1]["snapshot_id"] == target)

    def test_chain_error_on_dangling_parent(self) -> None:
        row = self.service.manager.store.list_snapshots(self.volume_name)[-1]
        self.service.manager.store.conn.execute(
            "UPDATE snapshots SET parent_id=? WHERE snapshot_id=?",
            ("missing-parent", row["snapshot_id"]),
        )
        self.service.manager.store.commit()
        with self.assertRaises(DomainError) as caught:
            self.service.manager.validate_chain(self.volume_name)
        self.assertEqual(caught.exception.code, E_CHAIN)
        self.assertEqual(caught.exception.snapshot_id, row["snapshot_id"])

    def test_audit_passes_on_healthy_chain_and_versions(self) -> None:
        verdict = self.service.audit(self.volume_name)
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["snapshot_count"], len(self.profile.snapshot_names))
        persisted = self.service.audit_verdict(self.volume_name)
        self.assertEqual(persisted["manifest_version"], verdict["manifest_version"])

    def test_audit_verdict_replaced_on_manifest_version_change(self) -> None:
        entries = self._export_entries()
        self.service.replace_manifest(self.volume_name, 9, entries)
        verdict = self.service.audit(self.volume_name)
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["manifest_version"], 9)
        self.assertEqual(self.service.audit_verdict(self.volume_name)["manifest_version"], 9)

    def _export_entries(self) -> list[dict]:
        entries = []
        for row in self.service.manager.store.list_snapshots(self.volume_name):
            refs = self.service.manager.store.ref_map(self.volume_name, row["snapshot_id"])
            entries.append({
                "snapshot_id": row["snapshot_id"],
                "name": row["name"],
                "version_no": row["version_no"],
                "parent_id": row["parent_id"],
                "bitmap_hex": bytes(row["bitmap"]).hex(),
                "refs": {str(b): h for b, h in refs.items()},
            })
        return entries


if __name__ == "__main__":
    unittest.main(verbosity=2)
