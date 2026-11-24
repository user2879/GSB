"""Optional Tkinter ledger audit viewer for Windows CPython."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .ledger import Ledger


def run_viewer(db_path: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise DiagnosticError(
            "E_CONFIG",
            "ui.tkinter",
            f"tkinter is unavailable: {exc}",
        ) from exc

    with Ledger(db_path) as ledger:
        batches = ledger.batches_as_dict()
        traces = ledger.traces_as_list()
        content_digest = ledger.content_digest
        format_version = ledger.format_version
        snapshot_version = ledger.snapshot_version
        session = ledger.recovery_status()

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - display dependent
        raise DiagnosticError("E_CONFIG", "ui.display", f"cannot open GUI: {exc}") from exc
    root.title("Receipt Diagnostics Ledger")
    root.geometry("1180x680")
    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(
        frame,
        text=(
            f"format_version={format_version}  snapshot_version={snapshot_version}  "
            f"session={session['session_state']}  batches={len(batches)}"
        ),
    ).pack(anchor=tk.W)
    ttk.Label(frame, text=f"content_digest={content_digest}").pack(anchor=tk.W, pady=(0, 8))

    columns = ("worker", "seq", "version", "requests", "param", "output", "receipt")
    table = ttk.Treeview(frame, columns=columns, show="headings", height=18)
    headings = {
        "worker": "Worker",
        "seq": "Batch",
        "version": "Ver",
        "requests": "Requests",
        "param": "Param digest",
        "output": "Output digest",
        "receipt": "Receipt digest",
    }
    widths = {
        "worker": 120,
        "seq": 70,
        "version": 60,
        "requests": 250,
        "param": 180,
        "output": 180,
        "receipt": 180,
    }
    for column in columns:
        table.heading(column, text=headings[column])
        table.column(column, width=widths[column], anchor=tk.W)
    for batch_id, batch in sorted(
        batches.items(), key=lambda item: (item[1]["worker_id"], item[1]["batch_seq"], item[0])
    ):
        table.insert(
            "",
            tk.END,
            iid=batch_id,
            values=(
                batch["worker_id"],
                batch["batch_seq"],
                batch["batch_version"],
                ", ".join(batch["request_ids"]),
                batch["param_digest"][:16] + "…",
                batch["output_digest"][:16] + "…",
                batch["receipt_digest"][:16] + "…",
            ),
        )
    table.pack(fill=tk.BOTH, expand=True)

    detail = tk.Text(frame, height=8, wrap=tk.WORD)
    detail.pack(fill=tk.X, pady=(8, 0))
    detail.insert(
        tk.END,
        f"Invalidation/recovery traces: {len(traces)}\n"
        f"Unknown batches: {len(session.get('unknown_batch_ids', []))}\n"
        f"Previous exit report available: {session.get('can_emit_previous_exit_report')}\n",
    )
    detail.configure(state=tk.DISABLED)
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tkinter receipt ledger viewer")
    parser.add_argument("--db", required=True)
    args = parser.parse_args(argv)
    try:
        run_viewer(args.db)
        return 0
    except DiagnosticError as exc:
        print(exc.as_dict(), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
