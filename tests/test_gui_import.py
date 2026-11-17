from __future__ import annotations

import unittest


class GuiImportTests(unittest.TestCase):
    def test_gui_module_import_is_lazy_and_headless_safe(self) -> None:
        import exp_tracker.gui as gui

        self.assertTrue(callable(gui.launch))


if __name__ == "__main__":
    unittest.main()
