from __future__ import annotations

import unittest


class GuiImportTests(unittest.TestCase):
    def test_gui_module_imports_without_creating_a_window(self) -> None:
        from batch_audit import gui

        self.assertTrue(callable(gui.main))


if __name__ == "__main__":
    unittest.main()
