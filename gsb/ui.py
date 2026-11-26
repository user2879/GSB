"""Optional Tkinter viewer for report.json.

The pipeline never imports this module during execution, so headless WSL/CI
environments can run without a display.  Launch with
``python -m gsb.cli ui --report artifacts/local-demo/report.json``.
"""

from __future__ import annotations

import json
from pathlib import Path


def launch(report_path: str | Path) -> None:
    import tkinter as tk
    from tkinter import ttk

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    root = tk.Tk()
    root.title("GSB 列存跳读诊断账本")
    root.geometry("1080x680")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    summary = report.get("summary", {})
    text = (
        f"session={report.get('session_id')}  spec={str(report.get('spec_digest'))[:16]}  "
        f"queries={summary.get('queries')}  false_skips={summary.get('false_skips')}  "
        f"false_retains={summary.get('false_retains')}  decoded_this_run={summary.get('decoded_block_groups_this_run')}"
    )
    ttk.Label(top, text=text).pack(anchor="w")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    issues_frame = ttk.Frame(notebook)
    attribution_frame = ttk.Frame(notebook)
    ledger_frame = ttk.Frame(notebook)
    notebook.add(issues_frame, text="Issues")
    notebook.add(attribution_frame, text="Attribution")
    notebook.add(ledger_frame, text="Ledger/Recovery")

    issues_tree = ttk.Treeview(issues_frame, columns=("query", "block", "code", "predicate", "missed"), show="headings")
    for column, width in (("query", 180), ("block", 120), ("code", 120), ("predicate", 360), ("missed", 80)):
        issues_tree.heading(column, text=column)
        issues_tree.column(column, width=width)
    for issue in report.get("issues", []):
        issues_tree.insert(
            "",
            "end",
            values=(
                issue.get("query_id"),
                issue.get("block_group_id"),
                issue.get("code"),
                json.dumps(issue.get("predicate"), ensure_ascii=False, sort_keys=True),
                issue.get("missed_rows", 0),
            ),
        )
    issues_tree.pack(fill="both", expand=True)

    attribution_tree = ttk.Treeview(
        attribution_frame,
        columns=("block", "query", "dominant", "index", "compression", "null_order"),
        show="headings",
    )
    for column in ("block", "query", "dominant", "index", "compression", "null_order"):
        attribution_tree.heading(column, text=column)
        attribution_tree.column(column, width=160)
    for entry in report.get("attribution", []):
        ratios = {item["source"]: item["ratio"] for item in entry.get("contributions", [])}
        attribution_tree.insert(
            "",
            "end",
            values=(
                entry.get("block_group_id"),
                entry.get("query_id"),
                entry.get("dominant_source"),
                ratios.get("index_distortion"),
                ratios.get("compression_mismatch"),
                ratios.get("predicate_null_order"),
            ),
        )
    attribution_tree.pack(fill="both", expand=True)

    ledger_text = tk.Text(ledger_frame, wrap="none")
    ledger_text.pack(fill="both", expand=True)
    ledger = report.get("ledger", {})
    recovery = ledger.get("recovery", {})
    ledger_text.insert("1.0", json.dumps({"ledger": {k: v for k, v in ledger.items() if k != "recovery"}, "recovery": recovery}, indent=2, sort_keys=True))
    ledger_text.configure(state="disabled")

    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    launch(args.report)
    return 0
