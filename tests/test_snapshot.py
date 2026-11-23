from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import clone_bundle, make_sample

from inpaint_pyramid.config import apply_command
from inpaint_pyramid.engine import IncrementalEngine, compare_engines
from inpaint_pyramid.snapshot import SnapshotManager


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.samples = Path(cls.temp.name) / "samples"
        cls.bundle, cls.stream, _ = make_sample(cls.samples, budget=3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_snapshot_version_increment_reuse_and_orphan_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manager = SnapshotManager(Path(raw) / "state")
            engine = IncrementalEngine(clone_bundle(self.bundle, budget=999))
            engine.commands_hash = self.stream.commands_hash
            engine.bootstrap()
            first = manager.commit(engine, {"commands": []})
            self.assertTrue(first["committed"])
            first_version = first["version"]
            engine.process_command(self.stream.commands[0], 0)
            second = manager.commit(engine, {"commands": [{"index": 0}]})
            self.assertGreater(second["version"], first_version)
            reused = manager.commit(engine, {"commands": [{"index": 0}]})
            self.assertFalse(reused["committed"])
            self.assertTrue(reused["reused_current"])
            self.assertEqual(reused["version"], second["version"])
            orphan = manager.snapshots_dir / ".tmp-forced-kill"
            orphan.mkdir()
            (orphan / "COMPLETE").write_text("{}", encoding="utf-8")
            info = manager.recover_info()
            self.assertEqual(info["recovered_version"], second["version"])
            self.assertEqual(info["command_cursor"], 0)

    def test_pending_snapshot_roundtrip_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manager = SnapshotManager(Path(raw) / "state")
            bundle = clone_bundle(self.bundle, budget=3)
            engine = IncrementalEngine(bundle)
            engine.commands_hash = self.stream.commands_hash
            engine.bootstrap()
            full_mutable = bundle.new_mutable()
            pending_seen = False
            for index in range(3):
                command = self.stream.commands[index]
                engine.process_command(command, index)
                try:
                    apply_command(full_mutable, command, bundle.spec)
                except Exception:
                    pass
                pending_seen = pending_seen or bool(engine.dirty_tiles)
            self.assertTrue(pending_seen)
            manager.commit(engine, {"commands": [{"index": 0}]})
            restored, manifest = manager.load(bundle, commands_hash=self.stream.commands_hash)
            self.assertEqual(restored.cursor, 2)
            self.assertEqual(restored.dirty_tiles, engine.dirty_tiles)
            self.assertEqual(restored.valid_tiles, engine.valid_tiles)
            for level in range(bundle.spec.levels):
                for key in restored.valid_tiles:
                    if key[0] == level:
                        x0, y0, x1, y1 = bundle.spec.tile_bounds(*key)
                        self.assertTrue(
                            (restored.layer_outputs[level][y0:y1, x0:x1] == engine.layer_outputs[level][y0:y1, x0:x1]).all()
                        )
            restored.drain_pending()
            self.assertFalse(restored.dirty_tiles)
            full = IncrementalEngine.build_full(
                bundle,
                mutable=full_mutable.clone(),
                cursor=2,
                commands_hash=self.stream.commands_hash,
            )
            self.assertTrue(compare_engines(restored, full)["equal"])


if __name__ == "__main__":
    unittest.main()
