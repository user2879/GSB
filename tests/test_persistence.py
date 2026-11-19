from __future__ import annotations

import json
import unittest

from forensic_workbench.errors import ForensicError
from forensic_workbench.pipeline import execute
from forensic_workbench.source import SourceSpec, generate_source
from forensic_workbench.state import SnapshotStore
from forensic_workbench.util import atomic_write_json
from tests.support import TemporaryWorkspace, make_workspace_source


class PersistenceTests(unittest.TestCase):
    def test_snapshot_roundtrip_and_stale_base_version(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            config = source.default_pipeline_config()
            result = execute(workspace, config, verify_full=True)
            default_run_dir = workspace / "runs" / "run-000001"
            self.assertTrue((default_run_dir / "close.json").exists())
            self.assertTrue((default_run_dir / "exit_manifest.json").exists())
            reopened = SnapshotStore(workspace).load()
            self.assertIsNotNone(reopened)
            self.assertEqual(reopened.snapshot_hash, result.commit.snapshot_hash)
            self.assertEqual(reopened.graph.node_hashes(), result.graph.node_hashes())
            self.assertEqual(
                reopened.graph.nodes["conclusion"].content_hash,
                result.conclusion_hash,
            )
            with self.assertRaises(ForensicError) as raised:
                execute(
                    workspace,
                    config,
                    expected_base_version=999,
                    verify_full=False,
                )
            self.assertEqual(raised.exception.code, "E_SNAPSHOT_VERSION")
            self.assertEqual(raised.exception.context["current_version"], 1)

    def test_snapshot_hash_conflict_returns_business_code(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            execute(workspace, source.default_pipeline_config(), verify_full=False)
            store = SnapshotStore(workspace)
            current = store.load()
            self.assertIsNotNone(current)
            value = json.loads(current.path.read_text(encoding="utf-8"))
            value["graph"]["param_versions"]["threshold"] = "tampered"
            current.path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ForensicError) as raised:
                store.load()
            self.assertEqual(raised.exception.code, "E_SNAPSHOT_VERSION")

    def test_orphans_are_quarantined_and_reported(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            result = execute(workspace, source.default_pipeline_config(), verify_full=False)
            store = SnapshotStore(workspace)
            snapshot = store.load()
            self.assertIsNotNone(snapshot)
            orphan_snapshot = store.snapshot_dir / "snapshot-999999-deadbeef.json"
            orphan_snapshot.write_text('{"uncommitted":true}\n', encoding="utf-8")
            pending = store.pending_dir / "pending-run.json"
            pending.write_text('{"pending":true}\n', encoding="utf-8")
            recovery = store.recover_orphans()
            self.assertFalse(orphan_snapshot.exists())
            self.assertFalse(pending.exists())
            self.assertTrue(
                all(issue["code"] == "E_SNAPSHOT_VERSION" for issue in recovery["report"]["issues"])
            )
            self.assertEqual(len(recovery["report"]["orphaned"]), 2)
            current = store.load()
            self.assertEqual(current.snapshot_hash, result.commit.snapshot_hash)

    def test_changed_source_does_not_reuse_old_cache(self):
        with TemporaryWorkspace() as workspace:
            spec = SourceSpec(
                width=160,
                height=90,
                fps=30,
                duration_seconds=3,
                seed=7,
                gop_size=10,
            )
            source = generate_source(workspace / "source", spec)
            config = source.default_pipeline_config()
            first = execute(workspace, config, verify_full=True)
            changed_spec = SourceSpec(
                width=160,
                height=90,
                fps=30,
                duration_seconds=3,
                seed=8,
                gop_size=10,
            )
            generate_source(workspace / "source", changed_spec, overwrite=True)
            second = execute(workspace, config, verify_full=True)
            self.assertEqual(second.audit["reused_nodes"], [])
            self.assertGreater(len(second.audit["recomputed_nodes"]), 0)
            self.assertNotEqual(first.conclusion_hash, second.conclusion_hash)

    def test_video_tamper_with_unchanged_manifest_is_rejected(self):
        with TemporaryWorkspace() as workspace:
            source = make_workspace_source(workspace)
            payload = bytearray(source.video_path.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            source.video_path.write_bytes(payload)
            with self.assertRaises(ForensicError) as raised:
                execute(workspace, source.default_pipeline_config(), verify_full=False)
            self.assertEqual(raised.exception.code, "E_FRAME_FORMAT")
            self.assertIn("source video hash mismatch", raised.exception.message)


if __name__ == "__main__":
    unittest.main()
