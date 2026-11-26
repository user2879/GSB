import json
import tempfile
import unittest
from pathlib import Path

from fxrender.app import build_engine
from fxrender.codes import BizCode
from fxrender.config import RenderConfig
from fxrender.interactive import handle_edit
from fxrender.pipeline import render_all
from fxrender.samplegen import generate_sample_bundle
from fxrender.snapshot import Snapshot, SnapshotStore


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp())
        self.bundle = generate_sample_bundle(
            self.work / "samples", frame_count=3)

    def _engine(self):
        return build_engine(self.bundle, self.work / "cache",
                            max_workers=1)[0]

    def test_atomic_commit_and_recover(self):
        engine = self._engine()
        result = render_all(engine)
        store = SnapshotStore(self.work / "snap")
        snapshot = Snapshot(
            global_version=engine.state.version,
            config_hash=engine.state.config_hash(),
            state_fp=engine.state.state_fp(),
            scheduler_state=getattr(engine, "_last_scheduler_state", {}),
            completed_frames=result.completed_frames,
            completed_tiles={k: list(v)
                             for k, v in result.completed_tiles.items()},
            consumed_budget=result.consumed_budget,
            pending_items=result.pending_items,
            failures=result.failures,
            cache_root=str(self.work / "cache"))
        ok, code = store.write_staged(
            snapshot, engine.state.version, engine.state.config_hash())
        self.assertTrue(ok)
        self.assertEqual(code, BizCode.OK.value)
        loaded, notes = store.recover(engine.state)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.completed_frames, result.completed_frames)
        self.assertTrue(notes)

    def test_version_mismatch_rejects_commit(self):
        engine = self._engine()
        render_all(engine)
        store = SnapshotStore(self.work / "snap2")
        snapshot = Snapshot(
            global_version=engine.state.version + 1,
            config_hash="stale",
            state_fp="x", scheduler_state={}, completed_frames=[],
            completed_tiles={}, consumed_budget=0, pending_items=[],
            failures=[], cache_root="")
        ok, code = store.write_staged(
            snapshot, engine.state.version, engine.state.config_hash())
        self.assertFalse(ok)
        self.assertEqual(code, BizCode.E_SNAPSHOT_VERSION.value)
        self.assertFalse(store.live_path.exists())

    def test_recovery_uses_only_committed_snapshot(self):
        store = SnapshotStore(self.work / "snap3")
        engine = self._engine()
        snapshot, notes = store.recover(engine.state)
        self.assertIsNone(snapshot)
        self.assertTrue(any("cold start" in n for n in notes))

    def test_diagnostics_written(self):
        store = SnapshotStore(self.work / "snap4")
        store.write_diagnostics({"signal": 15, "flushed": True})
        payload = json.loads(store.diagnostic_path.read_text())
        self.assertEqual(payload["signal"], 15)


class InteractiveTests(unittest.TestCase):
    def test_summary_returns_completed_progress(self):
        work = Path(tempfile.mkdtemp())
        bundle = generate_sample_bundle(work / "samples", frame_count=2)
        engine, _ = build_engine(bundle, work / "cache", max_workers=1)
        command = {"op": "set_weight", "condition": "text", "value": 0.6}
        summary, invalidated, result = handle_edit(engine, command, None)
        self.assertIn(summary.code, (BizCode.OK.value,
                                     BizCode.E_STALE_SUMMARY.value))
        self.assertEqual(summary.completed_frames, result.completed_frames)
        self.assertTrue(0.0 <= summary.progress <= 1.0)

    def test_bad_command_returns_business_code(self):
        work = Path(tempfile.mkdtemp())
        bundle = generate_sample_bundle(work / "samples", frame_count=2)
        engine, _ = build_engine(bundle, work / "cache2", max_workers=1)
        summary, _, _ = handle_edit(
            engine, {"op": "set_denoise_steps", "value": 0}, None)
        self.assertEqual(summary.code, BizCode.E_POSITIVE_INT.value)


if __name__ == "__main__":
    unittest.main()
