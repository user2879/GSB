"""Optional Tkinter dashboard; degrades gracefully when Tk is unavailable."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def launch_dashboard(db_path: Path, spec_path: Path) -> None:
    try:
        import tkinter as tk  # raises ImportError in minimal/headless interpreters
    except ImportError as exc:
        raise RuntimeError(f"Tkinter unavailable in this interpreter: {exc}") from exc

    root = tk.Tk()
    root.title("Lease Middleware Dashboard")

    text = tk.Text(root, width=100, height=30, font=("Consolas", 10))
    text.pack(fill="both", expand=True)

    def refresh() -> None:
        text.delete("1.0", "end")
        if not Path(db_path).exists():
            text.insert("end", "No ledger yet. Run: python demo.py run\n")
            return
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        text.insert("end", "== Leases ==\n")
        for r in conn.execute("SELECT lease_id,state,format FROM lease_records "
                              "ORDER BY lease_id"):
            text.insert("end", f"{r['lease_id']:<40} {r['state']:<8} {r['format']}\n")
        text.insert("end", "\n== Cursors ==\n")
        for r in conn.execute("SELECT * FROM client_cursors ORDER BY client_id"):
            text.insert("end", f"{r['client_id']:<20} seen={r['seen_config_version']}\n")
        text.insert("end", "\n== Subscription view ==\n")
        for r in conn.execute("SELECT * FROM subscription_view "
                              "ORDER BY client_id,config_key"):
            text.insert("end", f"{r['client_id']:<14} {r['config_key']:<18} "
                               f"{r['state']:<10} seen={r['seen_config_version']}\n")
        conn.close()
        root.after(1000, refresh)

    refresh()
    root.mainloop()
