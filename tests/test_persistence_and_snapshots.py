from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from spectral_workspace.errors import DomainError
from spectral_workspace.persistence import PersistenceStore
from spectral_workspace.synth import demo_operations, generate_sample_scene
from spectral_workspace.workspace import open_workspace


class PersistenceAndSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = generate_sample_scene()
        self.operations = demo_operations(self.sample)

    def test_controlled_exit_marker_and_exact_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = open_workspace(directory)
            for operation in self.operations[:3]:
                engine.apply(operation)
            original_hash = engine.state_hash
            engine.close()
            closed = Path(directory) / "CLOSED.json"
            self.assertTrue(closed.exists())
            self.assertFalse((Path(directory) / "SESSION_OPEN.json").exists())

            reopened = open_workspace(directory)
            self.assertEqual(reopened.recovery.prior_exit_state, "controlled_exit")
            self.assertEqual(reopened.revision, 3)
            self.assertEqual(reopened.state_hash, original_hash)
            self.assertEqual(reopened.verify()["status"], "ok")
            reopened.close()

    def test_uncommitted_staging_is_orphaned_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = open_workspace(directory)
            for operation in self.operations[:2]:
                engine.apply(operation)
            stable_hash = engine.state_hash
            engine.apply(self.operations[2], commit=False)
            self.assertIsNotNone(engine.pending_transaction)

            recovered = open_workspace(directory)
            self.assertEqual(
                recovered.recovery.prior_exit_state,
                "forced_termination_recovered",
            )
            self.assertEqual(len(recovered.recovery.orphaned_transactions), 1)
            self.assertEqual(recovered.revision, 2)
            self.assertEqual(recovered.state_hash, stable_hash)
            orphan_dir = Path(directory) / "orphaned"
            self.assertEqual(len([path for path in orphan_dir.iterdir() if path.is_dir()]), 1)
            self.assertTrue((Path(directory) / "recovery" / "orphan-records.jsonl").exists())
            recovered.close()

    def test_snapshot_version_rules_old_rejected_newer_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace_a = root / "a"
            workspace_b = root / "b"
            old_bundle = root / "old_bundle"
            new_bundle = root / "new_bundle"

            engine_a = open_workspace(workspace_a)
            engine_a.apply(self.operations[0])
            engine_a.close()
            shutil.copytree(
                workspace_a / "snapshots" / "snap-000000000001", old_bundle
            )

            engine_b = open_workspace(workspace_b)
            for operation in self.operations[:2]:
                engine_b.apply(operation)
            state_b_hash = engine_b.state_hash
            engine_b.close()
            shutil.copytree(
                workspace_b / "snapshots" / "snap-000000000002", new_bundle
            )

            store_b = PersistenceStore(workspace_b)
            current_before = store_b.read_current_doc()
            with self.assertRaises(DomainError) as context:
                store_b.import_snapshot_bundle(old_bundle)
            self.assertEqual(context.exception.code, "E_SNAPSHOT_VERSION")
            self.assertEqual(store_b.read_current_doc(), current_before)

            store_a = PersistenceStore(workspace_a)
            engine_a2 = open_workspace(workspace_a)
            self.assertEqual(engine_a2.snapshot_version, 1)
            engine_a2.close()
            store_a.import_snapshot_bundle(new_bundle)
            current_after = store_a.read_current_doc()
            self.assertEqual(current_after["snapshot_version"], 2)
            self.assertEqual(current_after["state_sha256"], state_b_hash)

            no_change = store_a.import_snapshot_bundle(new_bundle)
            self.assertFalse(no_change["changed"])

            engine_imported = open_workspace(workspace_a)
            self.assertEqual(engine_imported.revision, 2)
            self.assertEqual(engine_imported.state_hash, state_b_hash)
            engine_imported.close()

            with self.assertRaises(DomainError) as old_context:
                store_a.import_snapshot_bundle(old_bundle)
            self.assertEqual(old_context.exception.code, "E_SNAPSHOT_VERSION")


if __name__ == "__main__":
    unittest.main()
