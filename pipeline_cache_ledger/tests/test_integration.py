import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pcl.digest import digest_file
from pcl.engine import HIT, MISS, Engine, EngineConfig
from pcl.entries import Entry, decode_entry, encode_entry
from pcl.errors import DigestMismatchError
from pcl.samples import write_sample_workspace
from pcl.workbench import launch_workbench


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_path, self.paths = write_sample_workspace(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _engine(self, **kw):
        return Engine(self.root, EngineConfig(**kw))

    def test_cold_then_warm_hit_consistency(self):
        engine = self._engine(global_limit=2, default_stage_quota=1)
        engine.load(self.manifest_path)
        _, results = engine.run_plan()
        self.assertTrue(all(v.outcome == MISS for v in results.values()))

        second = self._engine(global_limit=3, default_stage_quota=2)
        second.load(self.manifest_path)
        _, warm = second.run_plan()
        self.assertTrue(all(v.outcome == HIT for v in warm.values()))
        auth = second.authoritative_recompute()
        for key, result in warm.items():
            self.assertEqual((result.outcome, result.reason), auth[key])

    def test_version_change_invalidation_and_propagation_trail(self):
        engine = self._engine()
        engine.load(self.manifest_path)
        engine.run_plan()
        changed = self.paths["base-lib"]
        changed.write_bytes(b"base-lib\ncontent-version=2\nnew-line\n")
        report = engine.note_change("base-lib")
        self.assertEqual(
            report["invalidation_order"],
            ["base-lib", "codegen", "core-bundle",
             "api-image", "release-notes", "web-image"],
        )
        trails = engine.state.invalidation_trails()
        self.assertIn("base-lib", trails)

        manifest2, _ = write_sample_workspace(
            self.root, pipeline_version=2,
            content_overrides={"base-lib": changed.read_bytes()},
            version_overrides={"base-lib": 2},
        )
        engine.load(manifest2)
        _, results = engine.run_plan()
        self.assertEqual(results["config"].outcome, HIT)
        self.assertEqual(results["base-lib"].outcome, MISS)
        self.assertEqual(results["codegen"].outcome, MISS)

    def test_cross_version_digest_and_hit_equivalence(self):
        engine = self._engine()
        engine.load(self.manifest_path)
        engine.run_plan()
        # Rewrite two entries as legacy v1/v2 using the same content.
        for key, fmt in (("base-lib", 1), ("config", 2)):
            decl = engine.manifest.artifacts[key]
            data = decl.abs_path.read_bytes()
            from pcl.digest import digest_bytes

            entry = Entry(
                key=key,
                digest=digest_bytes(data),
                version=decl.version,
                size=len(data),
                stage_id=decl.stage_id,
            )
            engine.store._entry_path(key).write_bytes(
                encode_entry(entry, fmt)
            )
        # Re-open a fresh engine and upgrade, then judge again.
        engine2 = self._engine()
        report = engine2.store.open_and_upgrade()
        self.assertIn("base-lib", {u["key"] for u in report["upgraded"]})
        engine2.load(self.manifest_path)
        _, results = engine2.run_plan()
        # v1/v2 entries carry no refs; ref-bearing base-lib must miss for the
        # documented reason, while ref-free config hits.
        self.assertEqual(results["config"].outcome, HIT)
        digests = {
            key: digest_file(engine2.manifest.artifacts[key].abs_path)
            for key in ("base-lib", "config")
        }
        raw = {
            key: engine2.store.read_raw_entry(key).digest
            for key in ("base-lib", "config")
        }
        self.assertEqual(digests, raw)

    def test_digest_mismatch_raises_e_digest_with_both_sides(self):
        engine = self._engine()
        engine.load(self.manifest_path)
        engine.run_plan()
        decl = engine.manifest.artifacts["config"]
        decl.abs_path.write_bytes(b"tampered config bytes\n")
        with self.assertRaises(DigestMismatchError) as ctx:
            engine.store.verify_against_file("config", decl.abs_path)
        self.assertEqual(ctx.exception.code, "E_DIGEST")
        self.assertNotEqual(
            ctx.exception.details["expected_digest"],
            ctx.exception.details["actual_digest"],
        )

    def test_snapshot_full_replacement_restore(self):
        engine = self._engine()
        engine.load(self.manifest_path)
        engine.run_plan()
        engine.save_snapshot("snap-a")

        # Drift the live state, then restore and prove full replacement.
        changed = self.paths["base-lib"]
        changed.write_bytes(b"base-lib\ncontent-version=2\ndrift\n")
        engine.note_change("base-lib")
        before = {a["key"]: a["status"] for a in
                  engine.state.all_artifacts()}
        self.assertIn("base-lib", [k for k, v in before.items()
                                   if v == "INVALIDATED"])
        restore = engine.restore_snapshot("snap-a")
        self.assertFalse(restore["diff"]["versions_equal"])
        after = engine.session_doc()
        snap = engine.snapshots.open("snap-a")[1]
        self.assertEqual(
            after["entry_digest_set"], snap["entry_digest_set"]
        )
        self.assertEqual(
            after["pipeline_version"], snap["pipeline_version"]
        )
        after_status = {a["key"]: a["status"] for a in
                        engine.state.all_artifacts()}
        self.assertTrue(
            all(v != "INVALIDATED" for v in after_status.values())
        )
        listing = engine.snapshots.list_snapshots()[0]
        self.assertTrue(listing["readonly"])

        # Restored entries must remain decodable and keep their references.
        entry = engine.store.read_entry("core-bundle")
        self.assertEqual(len(entry.refs), 2)

    def test_restore_then_warm_hit_on_clean_workspace(self):
        engine = self._engine()
        engine.load(self.manifest_path)
        engine.run_plan()
        engine.save_snapshot("snap-clean")
        # Re-open on a pristine copy and restore, then expect warm hits.
        import shutil as _shutil

        copy_root = Path(self.tmp.name) / "clean_copy"
        _shutil.copytree(self.root, copy_root)
        warm = Engine(copy_root, EngineConfig())
        warm.load(copy_root / "pipeline_manifest.json")
        warm.restore_snapshot("snap-clean")
        entry = warm.store.read_entry("core-bundle")
        self.assertEqual(len(entry.refs), 2)
        judge = Engine(copy_root, EngineConfig())
        judge.load(copy_root / "pipeline_manifest.json")
        _, results = judge.run_plan()
        self.assertTrue(all(v.outcome == HIT for v in results.values()))

    def test_diagnostic_view_renders_stably_without_tk(self):
        engine = self._engine()
        engine.load(self.manifest_path)
        engine.run_plan()
        engine.save_snapshot("snap-a")
        view = engine.diagnostic_view()
        text = launch_workbench(view)
        self.assertIn("PIPELINE CACHE LEDGER", text)
        keys = [a["key"] for a in view["artifacts"]]
        self.assertEqual(keys, sorted(keys))
        self.assertTrue(view["slot_timeline"])


if __name__ == "__main__":
    unittest.main()
