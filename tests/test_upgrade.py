from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from pagecache.manifest import read_manifest
from pagecache.upgrade import SimulatedUpgradeCrash, upgrade_manifest
from tests.helpers import clear_tree, make_samples, version_paths


class UpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.data, cls.meta = make_samples(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        for backup in Path(cls.temp.name).rglob("*.pre-upgrade.backup"):
            backup.chmod(0o666)
        clear_tree(Path(cls.temp.name))
        cls.temp.cleanup()

    def _copy_version(self, version: int, name: str) -> Path:
        source = self.data / "versions" / f"v{version}"
        target = Path(self.temp.name) / name
        shutil.copytree(source, target)
        return target

    def test_v1_and_v2_upgrade_fill_fields_and_preserve_backup(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version):
                base = self._copy_version(version, f"upgrade-v{version}")
                manifest = base / "manifest.pcm"
                result = upgrade_manifest(
                    manifest,
                    pages_dir=self.data / "pages",
                    overflow_dir=base / "overflow",
                )
                self.assertFalse(result.already_current)
                self.assertFalse(result.resumed)
                upgraded = read_manifest(manifest)
                self.assertEqual(upgraded.header_version, 3)
                self.assertEqual(upgraded.format_versions, (3,))
                for entry in upgraded.records:
                    self.assertIsNotNone(entry.payload_checksum)
                    self.assertTrue(entry.column_summaries)
                    self.assertEqual(entry.overflow_ref, f"{entry.page_id}.seg")
                    self.assertFalse(entry.compressed)
                self.assertTrue(result.report_path and Path(result.report_path).exists())
                self.assertTrue(Path(result.backup_path).exists())
                self.assertEqual(Path(result.backup_path).stat().st_mode & 0o777, 0o444)
                if version == 1:
                    self.assertEqual(
                        set(result.filled_fields),
                        {"column_summaries", "payload_checksum", "compressed", "overflow_ref"},
                    )
                else:
                    self.assertEqual(set(result.filled_fields), {"compressed", "overflow_ref"})

    def test_crash_resume_matches_one_shot_upgrade(self) -> None:
        resume_base = self._copy_version(1, "resume")
        one_shot_base = self._copy_version(1, "one-shot")
        resume_manifest = resume_base / "manifest.pcm"
        with self.assertRaises(SimulatedUpgradeCrash) as caught:
            upgrade_manifest(
                resume_manifest,
                pages_dir=self.data / "pages",
                overflow_dir=resume_base / "overflow",
                crash_after_checkpoint=2,
            )
        self.assertEqual(caught.exception.completed_records, 2)
        checkpoint_path = resume_manifest.with_name(f"{resume_manifest.name}.upgrade-checkpoint.json")
        self.assertTrue(checkpoint_path.exists())
        partial = read_manifest(resume_manifest)
        self.assertEqual(partial.format_versions, (1,))

        resumed = upgrade_manifest(resume_manifest, pages_dir=self.data / "pages", overflow_dir=resume_base / "overflow")
        one_shot = upgrade_manifest(
            one_shot_base / "manifest.pcm",
            pages_dir=self.data / "pages",
            overflow_dir=one_shot_base / "overflow",
        )
        self.assertTrue(resumed.resumed)
        self.assertFalse(one_shot.resumed)
        self.assertEqual(resumed.processed_record_indices, list(range(8)))
        self.assertEqual(len(resumed.processed_record_indices), len(set(resumed.processed_record_indices)))
        self.assertEqual(resume_manifest.read_bytes(), (one_shot_base / "manifest.pcm").read_bytes())
        self.assertEqual(
            hashlib.sha256(resume_manifest.read_bytes()).hexdigest(),
            hashlib.sha256((one_shot_base / "manifest.pcm").read_bytes()).hexdigest(),
        )

    def test_completed_checkpoint_is_idempotent(self) -> None:
        base = self._copy_version(2, "idempotent")
        manifest = base / "manifest.pcm"
        first = upgrade_manifest(manifest, pages_dir=self.data / "pages", overflow_dir=base / "overflow")
        second = upgrade_manifest(manifest, pages_dir=self.data / "pages", overflow_dir=base / "overflow")
        self.assertTrue(first.entries)
        self.assertTrue(second.already_current)
        self.assertEqual(read_manifest(manifest).format_versions, (3,))


if __name__ == "__main__":
    unittest.main()
