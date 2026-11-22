"""Small Tkinter diagnostics UI with read-only refresh semantics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .diagnostics import build_diagnostics
from .store import Ledger


class TkUnavailable(RuntimeError):
    pass


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def run_ui(
    database_path: str | os.PathLike[str],
    workdir: str | os.PathLike[str] | None = None,
) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except (ImportError, OSError) as exc:
        raise TkUnavailable(
            "当前 Python 缺少可加载的 Tk 运行时；请使用带 Tk 的 Windows Python 启动 UI"
        ) from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise TkUnavailable(
            "当前会话无法初始化图形显示；请在有桌面会话的 Windows Python 中启动 UI"
        ) from exc
    root.title("Sequential Transaction Message Ledger")
    root.geometry("1180x680")
    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)

    summary = tk.StringVar(value="loading")
    summary_label = ttk.Label(frame, textvariable=summary, justify="left")
    summary_label.pack(anchor="w", pady=(0, 6))

    partition_frame = ttk.LabelFrame(frame, text="Partitions", padding=4)
    partition_frame.pack(fill="both", expand=True)
    partition_tree = ttk.Treeview(
        partition_frame,
        columns=(
            "partition",
            "version",
            "max",
            "prefix",
            "truncated",
            "reason",
            "prepared",
            "committing",
            "committed",
            "aborted",
            "pending",
            "unknown",
        ),
        show="headings",
        height=8,
    )
    for column, heading, width in (
        ("partition", "Partition", 140),
        ("version", "Version", 65),
        ("max", "Max", 55),
        ("prefix", "Prefix", 60),
        ("truncated", "Truncated", 75),
        ("reason", "Reason", 90),
        ("prepared", "Prepared", 70),
        ("committing", "Committing", 80),
        ("committed", "Committed", 75),
        ("aborted", "Aborted", 65),
        ("pending", "Pending", 65),
        ("unknown", "Unknown", 130),
    ):
        partition_tree.heading(column, text=heading)
        partition_tree.column(column, width=width, anchor="w", stretch=True)
    partition_tree.pack(fill="both", expand=True)

    lower = ttk.Frame(frame)
    lower.pack(fill="both", expand=True, pady=(6, 0))
    unknown_frame = ttk.LabelFrame(lower, text="Unknown / rerun required", padding=4)
    unknown_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
    unknown_tree = ttk.Treeview(
        unknown_frame,
        columns=("transaction", "partition", "intervals"),
        show="headings",
        height=7,
    )
    for column, heading, width in (
        ("transaction", "Transaction", 190),
        ("partition", "Partition", 120),
        ("intervals", "Suspended positions", 230),
    ):
        unknown_tree.heading(column, text=heading)
        unknown_tree.column(column, width=width, anchor="w", stretch=True)
    unknown_tree.pack(fill="both", expand=True)

    snapshot_frame = ttk.LabelFrame(lower, text="Snapshots / backups", padding=4)
    snapshot_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
    snapshot_tree = ttk.Treeview(
        snapshot_frame,
        columns=("name", "format", "cursor", "readable"),
        show="headings",
        height=7,
    )
    for column, heading, width in (
        ("name", "Name", 190),
        ("format", "Ledger format", 100),
        ("cursor", "Cursor", 65),
        ("readable", "Readable", 75),
    ):
        snapshot_tree.heading(column, text=heading)
        snapshot_tree.column(column, width=width, anchor="w", stretch=True)
    snapshot_tree.pack(fill="both", expand=True)

    def clear(tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def refresh() -> None:
        try:
            with Ledger(database_path, auto_recover=False, readonly=True) as ledger:
                view = build_diagnostics(ledger, workdir)
            summary.set(
                "format={ledger_format_version} partition_set={partition_set_version} "
                "transaction_set={transaction_set_version} cursor={event_cursor}\n"
                "ledger_digest={ledger_digest}".format(**view)
            )
            clear(partition_tree)
            for row in view["partitions"]:
                distribution = row["state_distribution"]
                partition_tree.insert(
                    "",
                    "end",
                    values=(
                        row["partition_id"],
                        row["partition_version"],
                        row["max_position"],
                        row["prefix"],
                        row["truncated_position"],
                        row["truncated_reason"],
                        distribution["prepared"],
                        distribution["committing"],
                        distribution["committed"],
                        distribution["aborted"],
                        distribution["pending"],
                        ",".join(row["unknown_transaction_ids"]) or "-",
                    ),
                )
            clear(unknown_tree)
            for item in view["unknown_transactions"]:
                intervals = ",".join(
                    f"[{start},{end}]" for start, end in item["intervals"]
                )
                unknown_tree.insert(
                    "",
                    "end",
                    values=(item["transaction_id"], item["partition_id"], intervals),
                )
            clear(snapshot_tree)
            for item in view["snapshots"]:
                snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        item["name"],
                        item["ledger_format_version"],
                        item["event_cursor"],
                        item["readable"],
                    ),
                )
            for item in view["backups"]:
                snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        "backup: " + item["name"],
                        "-",
                        "-",
                        item["readonly"],
                    ),
                )
        except Exception as exc:
            messagebox.showerror("Diagnostics refresh failed", str(exc))
            summary.set("refresh failed: " + str(exc))

    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=(6, 0))
    ttk.Button(controls, text="Refresh (read-only)", command=refresh).pack(side="left")
    ttk.Button(controls, text="Close", command=root.destroy).pack(side="left", padx=6)
    refresh()
    root.mainloop()
