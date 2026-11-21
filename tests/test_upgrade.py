from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replica_repair.manifest_store import ManifestStore
from replica_repair.model import Manifest
from replica_repair.sample import init_sample, sample_manifest_path
from replica_repair.util import is_readonly
from tests.helpers import run_cli


class UpgradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        init_sample(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_v1_crash_resume_uses_committed_checkpoint_prefix(self) -> None:
        path = sample_manifest_path(self.root, "interop-v1")
        original = path.read_bytes()
        result = run_cli(
            ["upgrade", "--root", str(self.root), "--crash-after", "1", str(path)]
        )
        self.assertEqual(result.returncode, 86, result.stderr)
        self.assertEqual(Manifest.from_file(path).format_version, 1)
        backup = Path(str(path) + ".v1.readonly.bak")
        checkpoint = Path(str(path) + ".upgrade.checkpoint.json")
        self.assertTrue(backup.exists())
        self.assertTrue(is_readonly(backup))
        self.assertEqual(backup.read_bytes(), original)
        checkpoint_data = json.loads(checkpoint.read_text())
        self.assertEqual(len(checkpoint_data["processed_node_ids"]), 1)
        self.assertEqual(checkpoint_data["phase"], "upgrading")

        # If the processed record were touched again, this missing file would fail.
        (self.root / "nodes" / "v1-a" / "blocks" / "blk-0001.bin").unlink()
        resumed = ManifestStore(self.root).load_current(path)
        self.assertTrue(resumed.upgraded)
        self.assertTrue(resumed.resumed)
        self.assertEqual(resumed.skipped_committed, 1)
        self.assertEqual(resumed.processed_now, 2)
        self.assertEqual(resumed.manifest.format_version, 3)
        self.assertEqual(Manifest.from_file(path).format_version, 3)
        completed = json.loads(checkpoint.read_text())
        self.assertEqual(completed["phase"], "complete")
        self.assertEqual(len(completed["processed_node_ids"]), 3)
        self.assertTrue(is_readonly(backup))

        second_load = ManifestStore(self.root).load_current(path)
        self.assertFalse(second_load.upgraded)
        self.assertEqual(second_load.manifest.format_version, 3)

    def test_v2_upgrade_and_manifest_version_bump(self) -> None:
        path = sample_manifest_path(self.root, "interop-v2")
        source = Manifest.from_file(path)
        result = ManifestStore(self.root).load_current(path)
        self.assertTrue(result.upgraded)
        self.assertFalse(result.resumed)
        self.assertEqual(result.source_version, 2)
        self.assertEqual(result.manifest.format_version, 3)
        self.assertEqual(result.manifest.manifest_version, source.manifest_version + 1)
        backup = Path(str(path) + ".v2.readonly.bak")
        self.assertTrue(is_readonly(backup))
        self.assertEqual(Manifest.from_file(backup).format_version, 2)

    def test_checkpoint_without_backup_is_rejected(self) -> None:
        path = sample_manifest_path(self.root, "interop-v1")
        result = run_cli(["upgrade", "--root", str(self.root), "--crash-after", "1", str(path)])
        self.assertEqual(result.returncode, 86, result.stderr)
        Path(str(path) + ".v1.readonly.bak").unlink()
        with self.assertRaises(Exception) as caught:
            ManifestStore(self.root).load_current(path)
        self.assertEqual(getattr(caught.exception, "code", None), "E_STATE")


if __name__ == "__main__":
    unittest.main()
