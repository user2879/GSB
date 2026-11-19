from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from vecflow.gui import run_gui
from vecflow.logging_utils import JsonLogger


ROOT = Path(__file__).resolve().parents[1]


class GuiSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("tkinter") is not None, "Tkinter not available")
    @unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "Display not available")
    def test_gui_builds_and_renders_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = run_gui(
                data_path=str(ROOT / "samples" / "data_blocks.json"),
                tasks_path=str(ROOT / "samples" / "tasks.json"),
                ledger_path=str(Path(td) / "ledger.sqlite"),
                snapshot_dir=str(Path(td) / "snapshots"),
                logger=JsonLogger(Path(td) / "logs"),
                smoke=True,
                smoke_ms=50,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["timeline_rows"], 11)


if __name__ == "__main__":
    unittest.main()
