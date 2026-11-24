"""Optional Tkinter viewer.

Tkinter is part of the standard library on the target Windows Python.
Import is lazy so headless Linux boxes without ``libtk`` can still use
every non-GUI feature; attempting to open the viewer raises a clear error.
"""

from __future__ import annotations

import json
import os
import tempfile


def _import_tk():
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tkinter is unavailable in this Python environment; the CLI "
            f"(`python -m slowquery_diag.cli`) remains usable. Detail: {exc}"
        )
    return tk, ttk, filedialog, messagebox


class DiagnosticViewer:
    def __init__(self, declaration_path: str, workdir: str | None = None) -> None:
        tk, ttk, filedialog, messagebox = _import_tk()
        self._tk = tk
        self._ttk = ttk
        self._filedialog = filedialog
        self._messagebox = messagebox
        self.declaration_path = declaration_path
        self.workdir = workdir or tempfile.mkdtemp(prefix="sqdiag_gui_")
        self.root = tk.Tk()
        self.root.title("Slow-Query Diagnostic Viewer")
        self._build_layout()

    def _build_layout(self) -> None:
        tk = self._tk
        ttk = self._ttk
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.verdict_tab = ttk.Frame(self.notebook)
        self.bound_tab = ttk.Frame(self.notebook)
        self.replay_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.verdict_tab, text="Merged Verdicts")
        self.notebook.add(self.bound_tab, text="Fixed-Point Error Bounds")
        self.notebook.add(self.replay_tab, text="Replay Receipts")

        self.verdict_text = tk.Text(self.verdict_tab, wrap="none", width=110)
        self.verdict_text.pack(fill="both", expand=True)
        self.bound_text = tk.Text(self.bound_tab, wrap="none", width=110)
        self.bound_text.pack(fill="both", expand=True)
        self.replay_text = tk.Text(self.replay_tab, wrap="none", width=110)
        self.replay_text.pack(fill="both", expand=True)

        button_bar = ttk.Frame(self.root)
        button_bar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(button_bar, text="Reload Declaration",
                   command=self.reload).pack(side="left")
        ttk.Button(button_bar, text="Close",
                   command=self.root.destroy).pack(side="right")
        self.status = tk.StringVar(value="ready")
        ttk.Label(button_bar, textvariable=self.status).pack(side="left", padx=12)

    def reload(self) -> None:
        from .cli import _serialize
        from .engine import DiagnosticEngine
        ledger_path = os.path.join(self.workdir, "gui_ledger.db")
        catalog_path = os.path.join(self.workdir, "replay_catalog.json")
        try:
            engine = DiagnosticEngine(ledger_path, scale_exponent=1)
            result = engine.run(self.declaration_path)
            engine.build_replay_catalog(result.bundle, catalog_path)
            payload = _serialize(result)
            self.verdict_text.delete("1.0", "end")
            self.verdict_text.insert(
                "1.0", json.dumps(payload["verdicts"], indent=2,
                                  ensure_ascii=False, sort_keys=True)
            )
            self.bound_text.delete("1.0", "end")
            self.bound_text.insert(
                "1.0", json.dumps(payload["error_bounds"], indent=2,
                                  ensure_ascii=False, sort_keys=True)
            )
            self.replay_text.delete("1.0", "end")
            self.replay_text.insert(
                "1.0", json.dumps(payload["diagnosis"], indent=2,
                                  ensure_ascii=False, sort_keys=True)
            )
            self.status.set(f"scheme v{result.scheme_version} reloaded")
            engine.close()
        except Exception as exc:  # pragma: no cover - GUI feedback
            self._messagebox.showerror("Diagnostic error", str(exc))
            self.status.set("reload failed")

    def run(self) -> None:
        self.reload()
        self.root.mainloop()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Tkinter diagnostic viewer")
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--workdir")
    args = parser.parse_args()
    DiagnosticViewer(args.declaration, args.workdir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
