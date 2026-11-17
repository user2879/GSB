from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cluster_diag.runtime import DiagnosticRuntime
from cluster_diag.spec import load_spec

from .support import base_payload, free_ports, write_spec


class GuiSmokeTests(unittest.TestCase):
    def test_tk_console_queries_commits_and_recomputes(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            self.skipTest(f"tkinter unavailable: {exc}")
        try:
            root = tk.Tk()
        except Exception as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as temp_name:
                temp = Path(temp_name)
                path = write_spec(temp / "spec.json", base_payload(free_ports(2)))
                spec = load_spec(path)
                with DiagnosticRuntime(spec=spec, workdir=temp / "runtime", latency_budget_ms=50) as runtime:
                    runtime.commit_next()
                    from cluster_diag.gui import DiagnosticConsole

                    console = DiagnosticConsole(root, runtime, tk, ttk)
                    root.update_idletasks()
                    self.assertTrue(console.tree.exists("root"))
                    self.assertTrue(console.tree.exists("cluster:edge-a"))
                    console.tree.selection_set("cluster:edge-a")
                    console.query_selected()
                    root.update_idletasks()
                    self.assertIn("query edge-a", console.status_var.get())
                    console.commit_next()
                    root.update_idletasks()
                    self.assertIn("revision 2", console.status_var.get())
                    console.authoritative()
                    root.update_idletasks()
                    self.assertIn("authoritative OK", console.status_var.get())
                    console.recover()
                    root.update_idletasks()
                    self.assertIn("recovery OK", console.status_var.get())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
