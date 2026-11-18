from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quota_app.codec import peek_version
from quota_app.config_workspace import CHECKPOINT_FILENAME, open_workspace
from quota_app.errors import E_FORMAT, E_UPGRADE

from .helpers import make_legacy_workspace


class UpgradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.configs = make_legacy_workspace(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_upgrade_writes_backups_checkpoint_and_loads_all(self) -> None:
        result = open_workspace(self.configs)
        self.assertEqual(result.active.revision, 3)
        self.assertEqual(peek_version(self.configs / "config-v1.json"), 3)
        self.assertEqual(peek_version(self.configs / "config-v2.json"), 3)
        upgraded_names = {entry.filename for entry in result.upgrade.completed}
        self.assertEqual(upgraded_names, {"config-v1.json", "config-v2.json"})
        checkpoint = json.loads((self.configs / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["entries"]["config-v1.json"]["status"], "completed")
        self.assertEqual(checkpoint["entries"]["config-v2.json"]["status"], "completed")
        backup_1 = self.configs / "backups" / "config-v1.v1.bak"
        backup_2 = self.configs / "backups" / "config-v2.v2.bak"
        self.assertTrue(backup_1.exists())
        self.assertTrue(backup_2.exists())
        if os.name != "nt":
            self.assertFalse(backup_1.stat().st_mode & stat.S_IWUSR)
            self.assertFalse(backup_2.stat().st_mode & stat.S_IWUSR)
        # The upgraded snapshots expose their source version for the UI.
        v1_doc = next(doc for doc in result.documents if doc.source_path.endswith("config-v1.json"))
        self.assertEqual(v1_doc.original_version, 1)
        self.assertEqual(v1_doc.volumes[0].limit_dimension, "bytes")

    def test_reopen_does_not_reprocess_completed_entries(self) -> None:
        first = open_workspace(self.configs)
        self.assertEqual(len(first.upgrade.completed), 2)
        checkpoint_before = (self.configs / CHECKPOINT_FILENAME).read_bytes()
        second = open_workspace(self.configs)
        self.assertFalse(second.upgrade.completed)
        statuses = {entry.filename: entry.status for entry in second.upgrade.entries}
        self.assertEqual(statuses["config-v1.json"], "already_completed")
        self.assertEqual(statuses["config-v2.json"], "already_completed")
        self.assertEqual((self.configs / CHECKPOINT_FILENAME).read_bytes(), checkpoint_before)

    def test_forced_termination_after_first_file_then_resume(self) -> None:
        env = os.environ.copy()
        env["QUOTA_UPGRADE_KILL_AFTER"] = "1"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        command = [
            sys.executable,
            "-c",
            "from quota_app.config_workspace import open_workspace; open_workspace(r'%s')" % self.configs,
        ]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 86, completed.stderr)
        # File replacement happened, but the checkpoint was not updated yet.
        self.assertEqual(peek_version(self.configs / "config-v1.json"), 3)
        checkpoint = json.loads((self.configs / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
        self.assertNotIn("config-v1.json", checkpoint["entries"])

        resumed = open_workspace(self.configs)
        statuses = {entry.filename: entry.status for entry in resumed.upgrade.entries}
        self.assertEqual(statuses["config-v1.json"], "recovered_completion")
        self.assertEqual(statuses["config-v2.json"], "upgraded")
        checkpoint = json.loads((self.configs / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
        self.assertTrue(checkpoint["entries"]["config-v1.json"].get("recovered_after_interruption"))
        self.assertEqual(len(resumed.documents), 3)

    def test_unreadable_legacy_entry_is_e_upgrade_and_others_continue(self) -> None:
        (self.configs / "config-v1.json").write_text('{"format_version": 1,', encoding="utf-8")
        result = open_workspace(self.configs)
        codes = [error.code for error in result.errors]
        self.assertIn(E_UPGRADE, codes)
        entries = {entry.filename: entry for entry in result.upgrade.entries}
        self.assertEqual(entries["config-v1.json"].status, "failed")
        self.assertEqual(entries["config-v2.json"].status, "upgraded")
        self.assertEqual(result.active.revision, 3)

    def test_missing_checkpoint_blocks_legacy_upgrade_with_e_upgrade(self) -> None:
        checkpoint = self.configs / CHECKPOINT_FILENAME
        checkpoint.unlink()
        result = open_workspace(self.configs)
        self.assertTrue(any(error.code == E_UPGRADE for error in result.upgrade.errors))
        self.assertEqual(peek_version(self.configs / "config-v1.json"), 1)
        self.assertEqual(peek_version(self.configs / "config-v2.json"), 2)
        self.assertEqual(result.active.revision, 3)
        statuses = {entry.filename: entry.status for entry in result.upgrade.entries}
        self.assertEqual(statuses["config-v1.json"], "blocked")
        self.assertEqual(statuses["config-v2.json"], "blocked")

    def test_malformed_checkpoint_is_e_upgrade(self) -> None:
        (self.configs / CHECKPOINT_FILENAME).write_text("{not json", encoding="utf-8")
        result = open_workspace(self.configs)
        self.assertTrue(any(error.code == E_UPGRADE for error in result.upgrade.errors))
        self.assertEqual(peek_version(self.configs / "config-v1.json"), 1)

    def test_unknown_version_still_e_format(self) -> None:
        raw = json.loads((self.configs / "config-v3.json").read_text(encoding="utf-8"))
        raw["format_version"] = 8
        (self.configs / "config-v3.json").write_text(json.dumps(raw), encoding="utf-8")
        result = open_workspace(self.configs)
        # A current-format document with unknown version is a format failure.
        self.assertTrue(any(error.code == E_FORMAT for error in result.errors))


if __name__ == "__main__":
    unittest.main()
