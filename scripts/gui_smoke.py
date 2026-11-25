#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_ledger.gui import DiagnosticApp


def main() -> int:
    source = ROOT
    with tempfile.TemporaryDirectory(prefix="edge-ledger-gui-") as temporary:
        workspace = Path(temporary) / "workspace"
        for name in ("data", "config", "commands", "injections"):
            source_path = source / name
            if source_path.exists():
                shutil.copytree(source_path, workspace / name)
        master = tk.Tk()
        try:
            app = DiagnosticApp(master, workspace)
            master.update_idletasks()
            result = {
                "title": master.title(),
                "templates_rendered": len(app.template_tree.get_children()),
                "sample_rows": len(app.sample_tree.get_children()),
                "failure_rows": len(app.failure_tree.get_children()),
                "quick_view": app.quick_match_var.get(),
                "status": app.status_var.get(),
                "tk_version": tk.TkVersion,
            }
            app.threshold_var.set("850")
            app.apply_threshold()
            master.update_idletasks()
            result["edited_threshold"] = app._threshold()
            result["quick_view_after_edit"] = app.quick_match_var.get()
        finally:
            master.destroy()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["quick_view_after_edit"] == "Quick view: MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())

