from __future__ import annotations

import contextlib
import io
import unittest

from tunnelvpn.cli import main as cli_main


class CliUiSmokeTests(unittest.TestCase):
    def test_validate_cli(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli_main(["validate", "--config", "data/demo_routes.json"])
        self.assertEqual(code, 0)
        self.assertIn('"code": "OK"', output.getvalue())

    def test_ui_module_is_import_safe_without_tkinter(self) -> None:
        import tunnelvpn.ui as ui

        self.assertTrue(callable(ui.launch_ui))
        self.assertTrue(callable(ui.main))


if __name__ == "__main__":
    unittest.main()
