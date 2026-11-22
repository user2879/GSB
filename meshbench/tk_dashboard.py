"""Optional Tkinter dashboard (imported lazily; requires Tcl/Tk on Windows)."""

from .clock import mono_ms
from .view import build_view, render_headless


class DashboardApp:
    def __init__(self, config, ledger, refresh_ms=2000):
        import tkinter as tk
        from tkinter import ttk

        self.config = config
        self.ledger = ledger
        self.refresh_ms = refresh_ms
        self.root = tk.Tk()
        self.root.title("MeshBench — error budget workbench")
        self.text = tk.Text(self.root, width=110, height=34, wrap="none")
        self.text.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.root, command=self.text.yview)
        scrollbar.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scrollbar.set)
        self.status = tk.StringVar(value="starting...")
        ttk.Label(self.root, textvariable=self.status).pack(side="bottom", fill="x")
        self._refresh()

    def _refresh(self):
        started = mono_ms()
        view = build_view(self.config, self.ledger, self.config.version)
        elapsed = mono_ms() - started
        self.ledger.record_view_timing("tk_dashboard", started, elapsed)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", render_headless(view))
        self.status.set(
            f"cached view refresh took {elapsed}ms (monotonic clock; "
            f"truth source = ledger rows)"
        )
        self.root.after(self.refresh_ms, self._refresh)

    def mainloop(self):
        self.root.mainloop()
