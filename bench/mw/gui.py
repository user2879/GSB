"""Optional Tkinter dashboard for live partition progress.

Tkinter ships with standard Windows Python installs. In stripped-down
environments without tkinter, importing this module raises a clear,
actionable error while every headless capability keeps working.
"""
from __future__ import annotations

import sqlite3

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError as exc:  # pragma: no cover - environment dependent
    tk = None
    ttk = None
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None


class Dashboard:
    def __init__(self, ledger_path: str, log_path: str) -> None:
        if tk is None:
            raise RuntimeError(
                "tkinter is unavailable in this Python installation; use "
                "the headless CLI instead. On Debian/Ubuntu install "
                "python3-tk; standard Windows Python includes it.") from TK_IMPORT_ERROR
        self.ledger_path = ledger_path
        self.root = tk.Tk()
        self.root.title("Middleware Workbench - Partition Dashboard")
        columns = ("partition", "watermark_ms", "finalized_count",
                   "late_count", "late_seqs", "sum_fixed", "cursor_seq")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings")
        headers = {
            "partition": "Partition", "watermark_ms": "Watermark (ms)",
            "finalized_count": "Finalized", "late_count": "Late",
            "late_seqs": "Truncated late seqs", "sum_fixed": "Sum (fixed)",
            "cursor_seq": "Cursor seq"}
        for column in columns:
            self.tree.heading(column, text=headers[column])
            self.tree.column(column, width=140, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.status = tk.StringVar(value="starting...")
        ttk.Label(self.root, textvariable=self.status).pack(anchor="w",
                                                             padx=8, pady=4)
        self.refresh()

    def _current_version(self, conn: sqlite3.Connection) -> int | None:
        row = conn.execute(
            "SELECT MAX(version) AS v FROM param_versions").fetchone()
        return None if row[0] is None else row[0]

    def refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        conn = sqlite3.connect(self.ledger_path)
        try:
            version = self._current_version(conn)
            if version is not None:
                rows = conn.execute(
                    "SELECT partition,watermark_ms,finalized_count,"
                    "late_count,sum_fixed,cursor_seq FROM partition_state "
                    "WHERE param_version=? ORDER BY partition",
                    (version,)).fetchall()
                for row in rows:
                    late_rows = conn.execute(
                        "SELECT seq FROM late_log WHERE param_version=? "
                        "AND partition=? ORDER BY seq",
                        (version, row["partition"])).fetchall()
                    late_seqs = ",".join(str(r["seq"]) for r in late_rows)
                    self.tree.insert("", "end", values=(
                        row["partition"], row["watermark_ms"],
                        row["finalized_count"], row["late_count"],
                        late_seqs, row["sum_fixed"], row["cursor_seq"]))
                stale = conn.execute(
                    "SELECT COUNT(*) FROM param_versions "
                    "WHERE status='stale'").fetchone()[0]
                self.status.set(
                    f"param_version={version} | stale versions hidden: "
                    f"{stale} | refresh 1s")
        except sqlite3.OperationalError:
            self.status.set("ledger not initialized yet")
        finally:
            conn.close()
        self.root.after(1000, self.refresh)

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(ledger_path: str, log_path: str) -> None:
    Dashboard(ledger_path, log_path).run()
