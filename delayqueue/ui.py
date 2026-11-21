"""Tkinter diagnostic dashboard for a :class:`DelayQueue`.

Tkinter is imported lazily so the rest of the package works on machines
without a desktop session.  Every refresh reads :meth:`DelayQueue.snapshot`,
which is strictly read-only: rendering never changes the ledger.
"""

from __future__ import annotations


def _import_tk():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "tkinter is unavailable in this environment; on Windows use the "
            "official Python 3.10+ installer which bundles Tk/Tcl"
        ) from exc
    return tk, ttk


class DiagnosticDashboard:
    REFRESH_MS = 500

    def __init__(self, queue, title: str = "Delay Queue Diagnostics"):
        self.queue = queue
        self.title = title
        self._root = None

    def launch(self) -> None:
        tk, ttk = _import_tk()
        self._tk = tk
        root = tk.Tk()
        self._root = root
        root.title(self.title)
        root.geometry("860x620")

        self.meta_var = tk.StringVar()
        meta = ttk.Label(root, textvariable=self.meta_var,
                         font=("Consolas", 10, "bold"))
        meta.pack(anchor="w", padx=10, pady=(10, 4))

        columns = ("topic", "ready", "delayed", "inflight", "quota",
                   "lifetimes")
        table = ttk.Treeview(root, columns=columns, show="headings",
                             height=8)
        headings = {
            "topic": "Topic", "ready": "Ready", "delayed": "Delayed",
            "inflight": "Inflight", "quota": "Quota",
            "lifetimes": "Inflight lifetime (ms)",
        }
        widths = {"topic": 140, "ready": 80, "delayed": 90,
                  "inflight": 90, "quota": 80, "lifetimes": 320}
        for column in columns:
            table.heading(column, text=headings[column])
            table.column(column, width=widths[column], anchor="w")
        table.pack(fill="x", padx=10, pady=6)
        self.table = table

        self.attempts_var = tk.StringVar()
        ttk.Label(root, textvariable=self.attempts_var,
                  font=("Consolas", 10)).pack(anchor="w", padx=10, pady=4)

        ttk.Label(root, text="Dead letters (stable order):",
                  font=("Consolas", 10, "bold")).pack(
            anchor="w", padx=10, pady=(8, 2))
        dead_columns = ("message_id", "topic", "attempts",
                        "first_error", "last_error")
        dead = ttk.Treeview(root, columns=dead_columns, show="headings",
                            height=8)
        for column in dead_columns:
            dead.heading(column, text=column.replace("_", " ").title())
            dead.column(column, width=150 if column != "last_error" else 220,
                        anchor="w")
        dead.pack(fill="both", expand=True, padx=10, pady=6)
        self.dead = dead

        self._render()
        root.after(self.REFRESH_MS, self._schedule)
        root.mainloop()

    def _schedule(self) -> None:
        self._render()
        if self._root is not None and self._root.winfo_exists():
            self._root.after(self.REFRESH_MS, self._schedule)

    def _render(self) -> None:
        snapshot = self.queue.snapshot()
        self.meta_var.set(
            f"format v{snapshot['format_version']} | "
            f"ledger={snapshot['ledger_state']} | "
            f"spec={snapshot['spec_version']} | "
            f"dedup v{snapshot['dedup_version']} | "
            f"inflight {snapshot['global_inflight']}/"
            f"{snapshot['global_concurrency']} | "
            f"states={snapshot['state_counts']}")

        for item in self.table.get_children():
            self.table.delete(item)
        for topic in sorted(snapshot["topics"]):
            view = snapshot["topics"][topic]
            lifetimes = ", ".join(
                f"{message_id}:{age}ms"
                for message_id, age in sorted(
                    view["inflight_lifetime_ms"].items()))
            self.table.insert(
                "", "end",
                values=(topic, view["ready"], view["delayed"],
                        view["inflight"], view["quota"], lifetimes))

        self.attempts_var.set(
            "delivery attempt distribution: "
            + ", ".join(f"{attempts}:{count}" for attempts, count in
                        snapshot["attempt_distribution"].items()))

        for item in self.dead.get_children():
            self.dead.delete(item)
        for record in snapshot["dead"]:
            self.dead.insert("", "end", values=(
                record["message_id"], record["topic"], record["attempts"],
                record["first_error"], record["last_error"]))


def launch_dashboard(queue) -> None:
    DiagnosticDashboard(queue).launch()
