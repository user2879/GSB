"""Tkinter front-end with read-only refresh semantics."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..storage import Ledger


def _insert_tree(tree: ttk.Treeview, rows: list[tuple[str, ...]]) -> None:
    for item in tree.get_children():
        tree.delete(item)
    for row in rows:
        tree.insert("", "end", values=row)


class DiagnosticApp:
    def __init__(self, root: tk.Tk, db_path: str, session_id: str | None) -> None:
        self.root = root
        self.db_path = db_path
        self.session_id = session_id
        self.ledger = Ledger(db_path)
        self.root.title("Model Configuration Dispatch Workbench")
        self.root.geometry("1280x820")

        self.header = tk.StringVar()
        header = ttk.Label(root, textvariable=self.header, font=("Segoe UI", 11, "bold"))
        header.pack(fill="x", padx=10, pady=(8, 4))

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", padx=10)
        ttk.Button(toolbar, text="Refresh (read-only)", command=self.refresh).pack(side="left")
        ttk.Label(toolbar, text="Stable order: priority desc/id, rank/id, node/task/shard").pack(side="left", padx=12)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=8)
        self.task_tree = self._make_table(
            notebook,
            "Priority Queue",
            ("priority", "task", "node", "state", "completed", "wait_ms", "quota_ms", "error"),
            (70, 180, 90, 150, 170, 90, 100, 160),
        )
        self.preempt_tree = self._make_table(
            notebook,
            "Preemptions",
            ("id", "victim", "by", "requested", "suspended", "status", "resources"),
            (100, 170, 170, 90, 90, 180, 420),
        )
        self.decision_tree = self._make_table(
            notebook,
            "Decisions",
            ("id", "high", "victim", "verdict", "preemptable", "reason"),
            (90, 170, 170, 210, 100, 340),
        )
        self.receipt_tree = self._make_table(
            notebook,
            "Receipt Ledger",
            ("node", "task", "version", "shard", "generated", "decay", "score", "status", "output"),
            (90, 170, 80, 70, 90, 80, 80, 230, 300),
        )
        self.untrust_tree = self._make_table(
            notebook,
            "Node Untrust",
            ("rank", "node", "score", "untrusted", "receipts", "max_decay", "status_counts"),
            (60, 100, 80, 100, 90, 100, 360),
        )
        self.snapshot_tree = self._make_table(
            notebook,
            "Snapshots",
            ("snapshot", "label", "format", "digest", "created"),
            (230, 260, 80, 260, 220),
        )
        self.refresh()

    def _make_table(
        self,
        notebook: ttk.Notebook,
        title: str,
        columns: tuple[str, ...],
        widths: tuple[int, ...],
    ) -> ttk.Treeview:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for title_text, width in zip(columns, widths):
            tree.heading(title_text, text=title_text)
            tree.column(title_text, width=width, anchor="w")
        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def refresh(self) -> None:
        view = self.ledger.diagnostic_view(self.session_id)
        session = view.get("session")
        if session is None:
            self.header.set("No current non-stale session")
            return
        self.header.set(
            f"session {session['session_id']} | spec v{session['spec_version']} | "
            f"current config v{view['current_version']} | content {session['content_digest'][:16]}"
        )
        _insert_tree(
            self.task_tree,
            [
                (
                    str(task["priority"]),
                    task["task_id"],
                    task["node_id"],
                    task["state"],
                    ",".join(str(item["shard_index"]) for item in task.get("completed_shards", [])),
                    str(task["wait_ms"]),
                    str(task["used_quota_ms"]),
                    task["error_code"],
                )
                for task in view["priority_queue"]
            ],
        )
        _insert_tree(
            self.preempt_tree,
            [
                (
                    item["preemption_id"],
                    item["task_id"],
                    item["by_task_id"],
                    str(item["requested_shard"]),
                    str(item["suspended_shard"]),
                    item["status"],
                    str(item["resources"]),
                )
                for item in view["preemptions"]
            ],
        )
        _insert_tree(
            self.decision_tree,
            [
                (
                    item["decision_id"],
                    item["high_task_id"],
                    item["victim_task_id"],
                    item["verdict"],
                    str(item["preemptable"]),
                    item["reason"],
                )
                for item in view["decisions"]
            ],
        )
        _insert_tree(
            self.receipt_tree,
            [
                (
                    item["node_id"],
                    item["task_id"],
                    str(item["config_version"]),
                    str(item["shard_index"]),
                    str(item["generated_version"]),
                    str(item["decay_count"]),
                    str(item["untrust_score"]),
                    item["status"],
                    item["output_digest"],
                )
                for item in view["receipts"]
            ],
        )
        _insert_tree(
            self.untrust_tree,
            [
                (
                    str(item["rank"]),
                    item["node_id"],
                    str(item["score"]),
                    str(item["untrusted_receipts"]),
                    str(item["receipt_count"]),
                    str(item["max_decay"]),
                    str(item["status_counts"]),
                )
                for item in view["node_untrust"]
            ],
        )
        _insert_tree(
            self.snapshot_tree,
            [
                (
                    item["snapshot_id"],
                    item["label"],
                    str(item["format_version"]),
                    item["content_digest"],
                    item["created_at"],
                )
                for item in view["snapshots"]
            ],
        )


def launch(db_path: str, session_id: str | None = None) -> None:
    root = tk.Tk()
    app = DiagnosticApp(root, db_path, session_id)

    def close() -> None:
        app.ledger.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
