import unittest


class GuiImportTest(unittest.TestCase):
    def test_module_imports_and_declares_app(self) -> None:
        try:
            import importlib
            importlib.import_module("tkinter")
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"tkinter unavailable in this environment: {exc}")
        from k8ssim import gui
        self.assertTrue(hasattr(gui, "DiagnosticApp"))
        self.assertTrue(hasattr(gui, "main"))


if __name__ == "__main__":
    unittest.main()
