import json
import tempfile
import unittest
from pathlib import Path

from cache_invalidation.engine import Engine
from cache_invalidation.errors import EReplay
from cache_invalidation.model import EntryID
from cache_invalidation.snapshot import (
    DEFAULT_NAMESPACE,
    decode_snapshot_file,
    write_snapshot,
)
from cache_invalidation.store import Store
from tests.support import sample_graph


class RestoreReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _engine_with_history(self):
        engine = Engine(sample_graph(), store=Store(), format_version=3)
        engine.explicit_invalidate(EntryID("catalog", 100), 100)
        engine.explicit_invalidate(EntryID("catalog", 101), 200)
        return engine

    def test_v3_restore_replaces_state_and_replay_matches(self):
        engine = self._engine_with_history()
        snapshot_path = self.work / "v3.json"
        engine.save_snapshot(snapshot_path, 3)

        restored, data, report, replay_error = Engine.restore(
            snapshot_path, store=Store(self.work / "restored.sqlite"))
        self.assertIsNone(replay_error)
        self.assertTrue(report.replaced)
        self.assertEqual(data.format_version, 3)
        self.assertEqual(
            restored.versions[EntryID("search", 500)], 1)
        self.assertEqual(
            restored.versions[EntryID("catalog", 101)], 1)
        self.assertEqual(len(restored.records), 2)
        # 恢复目标是全新空库：不存在旧版本，因此没有版本差异条目。
        self.assertEqual(report.entry_diffs, [])
        self.assertEqual(report.replaced_edges, 6)
        self.assertEqual(report.replaced_namespaces, 5)

    def test_restore_is_full_replacement_not_incremental_merge(self):
        engine = self._engine_with_history()
        snapshot_path = self.work / "v3.json"
        engine.save_snapshot(snapshot_path, 3)

        existing_store = Store(self.work / "existing.sqlite")
        other = Engine(sample_graph(), store=existing_store)
        other.refresh_namespace("catalog", 50)
        other.save_snapshot(self.work / "other.json", 3)
        before_tables = existing_store.load_annotation_versions()
        self.assertTrue(before_tables)

        restored, _, report, replay_error = Engine.restore(
            snapshot_path, store=existing_store)
        self.assertIsNone(replay_error)
        self.assertEqual(report.replaced_namespaces, 5)
        self.assertEqual(restored.versions[EntryID("catalog", 100)], 1)
        # 旧会话状态已被整表替换，不与快照增量合并。
        self.assertEqual(
            existing_store.load_rounds()[0].clock_ms, 100)
        self.assertNotEqual(
            existing_store.load_rounds()[0].trigger_reason,
            "namespace_refresh")

    def test_replacement_report_lists_stable_entry_version_diffs(self):
        engine = self._engine_with_history()
        snapshot_path = self.work / "diffs.json"
        engine.save_snapshot(snapshot_path, 3)

        existing_store = Store(self.work / "diff_store.sqlite")
        other = Engine(sample_graph(), store=existing_store)
        other.explicit_invalidate(EntryID("catalog", 100), 50)

        _, _, report, replay_error = Engine.restore(
            snapshot_path, store=existing_store)
        self.assertIsNone(replay_error)
        diff_ids = [diff.entry_id for diff in report.entry_diffs]
        self.assertEqual(diff_ids, sorted(diff_ids))
        self.assertEqual(diff_ids, ["catalog/101"])
        sample = report.entry_diffs[0]
        self.assertEqual(sample.change, "version_changed")
        self.assertEqual(sample.old_version, 0)
        self.assertEqual(sample.new_version, 1)

    def test_v1_upgrade_reports_non_reproducible_replay(self):
        engine = self._engine_with_history()
        snapshot_path = self.work / "v1.json"
        write_snapshot(snapshot_path, engine._snapshot_graph(),
                       engine.records, 1)
        restored, data, report, replay_error = Engine.restore(
            snapshot_path, store=Store(self.work / "v1.sqlite"))
        self.assertEqual(data.format_version, 1)
        self.assertTrue(report.replaced)
        self.assertIsInstance(replay_error, EReplay)
        self.assertEqual(
            replay_error.details["basis"],
            "format_v1_lacks_annotation_sets")
        self.assertTrue(
            replay_error.key.startswith(f"{DEFAULT_NAMESPACE}/"))

    def test_replay_mismatch_locates_first_trigger_entry_and_field(self):
        engine = self._engine_with_history()
        snapshot_path = self.work / "tampered.json"
        engine.save_snapshot(snapshot_path, 3)
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload["rounds"][0]["affected"].remove("storefront/400")
        snapshot_path.write_text(
            json.dumps(payload), encoding="utf-8")

        _, _, _, replay_error = Engine.restore(
            snapshot_path,
            store=Store(self.work / "tampered.sqlite"))
        self.assertIsInstance(replay_error, EReplay)
        self.assertEqual(replay_error.key, "catalog/100")
        self.assertIn("affected", replay_error.field_path)
        self.assertEqual(
            replay_error.details["basis"], "affected_set_mismatch")

    def test_state_version_mismatch_still_triggers_replacement(self):
        engine = self._engine_with_history()
        snapshot_path = self.work / "state.json"
        write_snapshot(snapshot_path, engine._snapshot_graph(),
                       engine.records, 3, state_version=99)
        _, data, report, replay_error = Engine.restore(
            snapshot_path, store=Store(self.work / "state.sqlite"))
        self.assertEqual(data.state_version, 99)
        self.assertTrue(report.replaced)
        self.assertIsNone(replay_error)


if __name__ == "__main__":
    unittest.main()
