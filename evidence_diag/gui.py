from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path
from typing import Any


class Dashboard:
    def __init__(self, root: Path, tk: Any, ttk: Any) -> None:
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.window = tk.Tk()
        self.window.title("Evidence Chain Diagnostics")
        self.window.geometry("1100x720")
        self.text = tk.Text(self.window, wrap="word", font=("monospace", 10))
        scrollbar = ttk.Scrollbar(
            self.window, orient="vertical", command=self.text.yview
        )
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.refresh()

    def _report_path(self) -> Path:
        return self.root / "reports" / "demo_summary.json"

    def refresh(self) -> None:
        self.text.delete("1.0", "end")
        path = self._report_path()
        if not path.exists():
            self.text.insert(
                "end",
                "No demo report found.\n\nRun:\n  python -m evidence_diag demo\n",
            )
            return
        summary = json.loads(path.read_text(encoding="utf-8"))
        diagnostic_path = self.root / "reports" / "diagnostic.json"
        recovery_path = self.root / "reports" / "recovery.json"
        blocks = {
            "DEMO SUMMARY": summary,
            "DIAGNOSTIC": (
                json.loads(diagnostic_path.read_text(encoding="utf-8"))
                if diagnostic_path.exists()
                else {}
            ),
            "RECOVERY": (
                json.loads(recovery_path.read_text(encoding="utf-8"))
                if recovery_path.exists()
                else {}
            ),
        }
        for title, payload in blocks.items():
            self.text.insert("end", f"=== {title} ===\n")
            self.text.insert(
                "end", json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n"
            )

    def run(self) -> int:
        self.window.mainloop()
        return 0


def _bootstrap_local_tk(root: Path) -> dict[str, Any]:
    vendor = root / ".local" / "tk"
    lib_dir = vendor / "usr" / "lib" / "x86_64-linux-gnu"
    tcl_dir = vendor / "usr" / "share" / "tcltk" / "tcl8.6"
    tk_dir = vendor / "usr" / "share" / "tcltk" / "tk8.6"
    loaded: list[str] = []
    if platform.machine() in ("x86_64", "AMD64") and lib_dir.exists():
        for name in (
            "libXft.so.2",
            "libXss.so.1",
            "libtcl8.6.so",
            "libtk8.6.so",
        ):
            path = lib_dir / name
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(str(path))
        if tcl_dir.exists():
            os.environ.setdefault("TCL_LIBRARY", str(tcl_dir))
        if tk_dir.exists():
            os.environ.setdefault("TK_LIBRARY", str(tk_dir))
        old_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            value for value in (str(lib_dir), old_path) if value
        )
    return {
        "local_tk": bool(loaded),
        "preloaded": [Path(item).name for item in loaded],
        "tcl_library": os.environ.get("TCL_LIBRARY"),
        "tk_library": os.environ.get("TK_LIBRARY"),
    }


def run_gui(root: str | Path, *, smoke_test: bool = False) -> int:
    root_path = Path(root)
    bootstrap = _bootstrap_local_tk(root_path)
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise RuntimeError(
            "tkinter is unavailable; install python3-tk and retry. "
            f"Original error: {exc}"
        ) from exc
    try:
        dashboard = Dashboard(root_path, tk, ttk)
    except Exception as exc:
        raise RuntimeError(
            "Tk could not create a display; use the CLI demo in headless mode. "
            f"Original error: {exc}"
        ) from exc
    if smoke_test:
        version = dashboard.window.tk.call("info", "patchlevel")
        dashboard.window.withdraw()
        dashboard.window.update_idletasks()
        dashboard.window.destroy()
        print(
            json.dumps(
                {
                    "tk_smoke_test": "ok",
                    "tk_patchlevel": version,
                    "bootstrap": bootstrap,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    return dashboard.run()
