import json
import os
import tempfile
import unittest

from configmesh.errors import BusinessError, ErrorCode
from configmesh.injections import (inject_corrupt_manifest, inject_derived_cycle,
                                   illegal_grid_parameter, restore)
from configmesh.fixedpoint import Grid
from configmesh.manifest import load_bundle
from configmesh.preview import PreviewEngine
from configmesh.compiler import compile_version, build_dispatch_plan
from configmesh.storage import SnapshotStore
from configmesh.util import content_digest


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = os.path.join(self.tmp, "manifest.json")
        with open("examples/manifest.json", "rb") as src, open(self.manifest, "wb") as dst:
            dst.write(src.read())

    def test_corrupt_manifest_e_manifest_then_restore(self):
        descriptor = inject_corrupt_manifest(self.manifest)
        self.assertEqual(descriptor["expected_code"], "E_MANIFEST")
        registry, _ = load_bundle("examples/manifest.json", "examples/registry.json")
        from configmesh.manifest import load_manifest
        with self.assertRaises(BusinessError) as ctx:
            load_manifest(self.manifest, registry)
        self.assertEqual(ctx.exception.code, ErrorCode.E_MANIFEST)
        restore(self.manifest)
        versions = load_manifest(self.manifest, registry)
        self.assertEqual(len(versions), 3)

    def test_cycle_injection_e_cycle(self):
        inject_derived_cycle(self.manifest)
        registry, _ = load_bundle("examples/manifest.json", "examples/registry.json")
        from configmesh.manifest import load_manifest
        with self.assertRaises(BusinessError) as ctx:
            load_manifest(self.manifest, registry)
        self.assertEqual(ctx.exception.code, ErrorCode.E_CYCLE)

    def test_illegal_grid_e_grid(self):
        with self.assertRaises(BusinessError) as ctx:
            Grid.create(illegal_grid_parameter())
        self.assertEqual(ctx.exception.code, ErrorCode.E_GRID)


class SnapshotTests(unittest.TestCase):
    def test_digest_full_replacement_and_persistence(self):
        db_path = os.path.join(tempfile.mkdtemp(), "snap.sqlite")
        store = SnapshotStore(db_path)
        payload = {"a": 1, "plan": {"x": [1, 2]}}
        snap = store.save_snapshot("s", 1, 3, "CONVERGED", payload)
        self.assertEqual(snap.content_digest, content_digest(payload))
        self.assertTrue(store.identity_matches(snap, 1, 3, payload))
        self.assertFalse(store.identity_matches(snap, 2, 3, payload))
        removed = store.replace_all_session_state("s")
        self.assertEqual(removed, 1)
        self.assertIsNone(store.load_latest("s"))
        store.close()


class PreviewTests(unittest.TestCase):
    def test_preview_drift_first_field(self):
        registry, versions = load_bundle("examples/manifest.json", "examples/registry.json")
        grid = Grid.create(3)
        version = versions[0]
        plan = build_dispatch_plan(compile_version(version, grid), version)
        engine = PreviewEngine()
        engine.refresh(version, plan, grid)
        self.assertTrue(engine.verify_cache(version, plan, grid)["preview_consistent"])
        derived_key = next(k for k, item in version.items.items() if item.is_derived())
        engine.simulate_ui_drift(version.targets[0], 0, derived_key, 42)
        with self.assertRaises(BusinessError) as ctx:
            engine.verify_cache(version, plan, grid)
        self.assertEqual(ctx.exception.code, ErrorCode.E_PREVIEW)
        self.assertEqual(ctx.exception.details[0]["node_id"], version.targets[0])
        self.assertIn(derived_key, ctx.exception.details[0]["field"])


if __name__ == "__main__":
    unittest.main()


class ReplayMismatchTests(unittest.TestCase):
    def test_corrupted_snapshot_raises_e_replay_with_key_and_field(self):
        import tempfile
        from configmesh.session import Session
        workdir = tempfile.mkdtemp(prefix="configmesh-replay-")
        session = Session("replay-negative", workdir, grid_exp=3)
        try:
            session.load_inputs("examples/manifest.json", "examples/registry.json")
            session.start_nodes()
            outputs = [session.run_version(v) for v in session.versions]
            session.shutdown_nodes()
            version = session.versions[0]
            snapshot = session.store.load_version("replay-negative", version.version)
            # corrupt one stored artifact field directly in the SQLite payload
            first_node = version.targets[0]
            first_key = sorted(snapshot.payload["plan"]["nodes"][first_node]["items"])[0]
            snapshot.payload["plan"]["nodes"][first_node]["items"][first_key]["qvalue"] += 777
            session.store.save_snapshot(
                "replay-negative", version.version, 3, snapshot.state, snapshot.payload)
            with self.assertRaises(BusinessError) as ctx:
                session.replay_snapshot(version.version)
            self.assertEqual(ctx.exception.code, ErrorCode.E_REPLAY)
            first = ctx.exception.details[0]["first_mismatch"]
            self.assertEqual(first["key"], first_key)
            self.assertEqual(first["field"], "qvalue")
        finally:
            session.close()
