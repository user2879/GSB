from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from config_arbiter.diagnostics import build_diagnostic_model
from config_arbiter.util import canonical_bytes
from tests.support import build_engine


class ReadOnlyDiagnosticsTests(unittest.TestCase):
    def test_refresh_is_deterministic_and_does_not_change_ledger_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine, ledger, specs, _, _ = build_engine(Path(temporary))
            try:
                engine.run_all()
                digest_before = ledger.meta()["content_digest"]
                first = build_diagnostic_model(specs, ledger)
                digest_mid = ledger.meta()["content_digest"]
                second = build_diagnostic_model(specs, ledger)
                digest_after = ledger.meta()["content_digest"]
                self.assertEqual(digest_before, digest_mid)
                self.assertEqual(digest_mid, digest_after)
                self.assertEqual(canonical_bytes(first), canonical_bytes(second))
                self.assertEqual(
                    [item["environment_id"] for item in first["environments"]],
                    ["dr-ap", "prod-cn", "staging-eu"],
                )
                for view in first["views"]:
                    self.assertEqual([item["key"] for item in view["keys"]], sorted(item["key"] for item in view["keys"]))
            finally:
                ledger.close()

    def test_ui_module_imports_when_tk_runtime_is_available(self) -> None:
        try:
            import tkinter  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Tk runtime unavailable in this headless container: {exc}")
        module = importlib.import_module("config_arbiter.ui")
        self.assertTrue(hasattr(module, "ArbiterApp"))
        self.assertTrue(hasattr(module, "launch_gui"))


if __name__ == "__main__":
    unittest.main()
