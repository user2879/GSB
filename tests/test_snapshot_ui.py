from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from spectral_workbench.config import InjectionConfig
from spectral_workbench.errors import ErrorCode, WorkbenchError
from spectral_workbench.generator import generate_cube
from spectral_workbench.pipeline import PipelineRunner
from spectral_workbench.snapshot import SnapshotStore, load_snapshot, make_snapshot, save_snapshot
from spectral_workbench.ui import DiagnosticWindow
from tests.helpers import small_cube_config, small_pipeline_config


class SnapshotAndViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cube = Path(cls.temporary.name) / "cube"
        output = Path(cls.temporary.name) / "run"
        generate_cube(small_cube_config(), InjectionConfig(), cube)
        cls.result = PipelineRunner(cube, small_pipeline_config()).run(output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_snapshot_roundtrip_hash_and_immutability(self) -> None:
        snapshot = make_snapshot(self.result, version=1)
        directory = Path(self.temporary.name) / "snapshot_one"
        save_snapshot(snapshot, directory)
        loaded = load_snapshot(directory)
        self.assertEqual(loaded.content_hash, snapshot.content_hash)
        self.assertFalse(loaded.pseudocolor.flags.writeable)
        self.assertFalse(loaded.concentration_curves.flags.writeable)
        self.assertEqual(loaded.pseudocolor.ndim, 3)
        self.assertEqual(loaded.concentration_curves.shape[1], 8)

    def test_stale_snapshot_is_rejected(self) -> None:
        first = make_snapshot(self.result, version=1)
        second = make_snapshot(self.result, version=2)
        store = SnapshotStore()
        store.publish(first)
        store.publish(second)
        with self.assertRaises(WorkbenchError) as context:
            store.validate_render(first)
        self.assertEqual(context.exception.code, ErrorCode.E_SNAPSHOT_STALE)

    def test_view_renders_only_current_snapshot(self) -> None:
        app = QApplication.instance() or QApplication([])
        snapshot = make_snapshot(self.result, version=3)
        store = SnapshotStore()
        store.publish(snapshot)
        window = DiagnosticWindow()
        window.render_snapshot(snapshot, store)
        self.assertIn(snapshot.content_hash[:12], window.windowTitle())
        self.assertFalse(hasattr(window, "service"))
        self.assertFalse(hasattr(window, "pipeline"))
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
