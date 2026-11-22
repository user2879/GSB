import unittest

from snapshot_migration import ui


class UIAvailabilityTests(unittest.TestCase):
    def test_module_imports_even_without_tk(self):
        # The module must always import; only launch_ui depends on a display.
        self.assertTrue(hasattr(ui, "launch_ui"))
        if not ui.TK_AVAILABLE:
            with self.assertRaises(RuntimeError):
                ui.launch_ui("samples/snapshot_chain.json", ":memory:")
        else:
            self.assertTrue(callable(ui.DiagnosticUI))


if __name__ == "__main__":
    unittest.main()
