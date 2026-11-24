from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .view import authoritative_rows, interactive_view


def launch_ui(report_path: str | Path | None = None) -> int:
    import tkinter as tk
    from tkinter import ttk

    report: dict[str, Any] = {}
    if report_path and Path(report_path).exists():
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    root = tk.Tk()
    root.title("Batch Receipt Ledger")
    root.geometry("1180x720")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    status_var = tk.StringVar(value="load a report or run demo")
    progress_var = tk.StringVar(value="progress: -")
    invalid_var = tk.StringVar(value="invalid: -")
    ttk.Label(top, textvariable=status_var, font=("Segoe UI", 11, "bold")).pack(side="left")
    ttk.Label(top, textvariable=progress_var).pack(side="left", padx=16)
    ttk.Label(top, textvariable=invalid_var).pack(side="left")

    columns = ("request", "batch", "worker", "authoritative", "basis", "duplicate", "invalid")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    headings = {
        "request": "Request",
        "batch": "Batch",
        "worker": "Worker",
        "authoritative": "Authoritative Ref",
        "basis": "Basis",
        "duplicate": "Dup",
        "invalid": "Invalid",
    }
    widths = {"request": 110, "batch": 100, "worker": 90, "authoritative": 190, "basis": 280, "duplicate": 50, "invalid": 70}
    for column in columns:
        tree.heading(column, text=headings[column])
        tree.column(column, width=widths[column], anchor="w")
    tree.pack(fill="both", expand=True, padx=8)

    details = tk.Text(root, height=14, wrap="word")
    details.pack(fill="both", expand=False, padx=8, pady=8)

    def render_rows(rows: list[dict[str, Any]]) -> None:
        for item in tree.get_children():
            tree.delete(item)
        for row in rows:
            tree.insert(
                "",
                "end",
                iid=row["request_id"],
                values=(
                    row["request_id"],
                    row["authoritative_batch_id"],
                    row["authoritative_worker_id"],
                    row["authoritative_ref"],
                    row["basis"],
                    "yes" if row["is_duplicate"] else "no",
                    "yes" if row["invalidated"] else "no",
                ),
            )

    def refresh_fast() -> None:
        if not report:
            status_var.set("No report loaded")
            return
        all_rows = authoritative_rows(report)
        completed = [row["request_id"] for row in all_rows[: max(1, len(all_rows) // 2)]]
        result = interactive_view(report, completed_request_ids=completed, budget_ms=250)
        render_rows([row for row in all_rows if row["request_id"] in set(completed)])
        summary = result["summary"]
        status_var.set(f"FAST: {result['status']}")
        progress_var.set(f"progress: {summary['completed_entries']}/{summary['total_entries']}")
        invalid_var.set(f"current invalid: {summary['invalid_entry_count']}")
        details.delete("1.0", "end")
        details.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    def refresh_authoritative() -> None:
        if not report:
            status_var.set("No report loaded")
            return
        rows = authoritative_rows(report)
        render_rows(rows)
        result = interactive_view(report, budget_ms=10_000)
        status_var.set(f"AUTHORITATIVE: {result['status']}")
        summary = result["summary"]
        progress_var.set(f"progress: {summary['completed_entries']}/{summary['total_entries']}")
        invalid_var.set(f"invalid: {summary['invalid_entry_count']}")
        details.delete("1.0", "end")
        details.insert("1.0", json.dumps(report.get("invalidation", {}), ensure_ascii=False, indent=2, sort_keys=True))

    buttons = ttk.Frame(root, padding=(8, 0, 8, 8))
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Fast conclusion", command=refresh_fast).pack(side="left")
    ttk.Button(buttons, text="Authoritative recompute", command=refresh_authoritative).pack(side="left", padx=8)

    if report:
        refresh_authoritative()
    root.mainloop()
    return 0
