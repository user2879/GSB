from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class GuiSmokeTests(unittest.TestCase):
    def test_window_reads_persisted_snapshots(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from mesh_audit.gui import MainWindow
        from mesh_audit.sample import deterministic_mesh_payload, deterministic_repair_script
        from tests.helpers import run_audit

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mesh_path = root / "mesh.json"
            script_path = root / "script.json"
            mesh_path.write_text(json.dumps(deterministic_mesh_payload()), encoding="utf-8")
            script_path.write_text(json.dumps(deterministic_repair_script()), encoding="utf-8")
            run_audit(root)
            window = MainWindow(
                str(mesh_path),
                str(script_path),
                str(root / "state"),
                str(root / "output"),
            )
            window.show()
            app.processEvents()
            self.assertEqual(len(window.workspace.steps), 5)
            self.assertTrue(all(step.valid for step in window.workspace.steps))
            self.assertIsNotNone(window.original_canvas.mesh)
            self.assertIsNotNone(window.current_canvas.mesh)
            self.assertGreater(len(window.current_canvas._projected), 0)
            self.assertGreater(window.codes_list.count(), 0)
            self.assertGreater(window.vertex_table.rowCount(), 0)
            self.assertGreater(window.propagation_table.rowCount(), 0)
            window.close()


if __name__ == "__main__":
    unittest.main()
