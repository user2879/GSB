from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .errors import EnvironmentError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CloudShard local ledger viewer")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--smoke", action="store_true", help="build the window and close it immediately")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise EnvironmentError(
            "Tkinter is unavailable. On Ubuntu/WSL install system package 'sudo apt install python3-tk tk8.6' "
            "(or python3.11-tk if using Python 3.11 from apt), then retry cloudshard tk.",
            field="tkinter",
        ) from exc

    ledger = Path(args.ledger)
    if not ledger.exists():
        raise CloudShardError(f"ledger does not exist: {ledger}", field="ledger")

    root = tk.Tk()
    root.title("CloudShard Ledger Viewer")
    root.geometry("1080x680")
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text=f"Ledger: {ledger}", font=("monospace", 10)).pack(side="left")

    body = ttk.Panedwindow(root, orient="horizontal")
    body.pack(fill="both", expand=True, padx=8, pady=8)
    list_frame = ttk.Frame(body)
    detail_frame = ttk.Frame(body)
    body.add(list_frame, weight=1)
    body.add(detail_frame, weight=3)

    columns = ("shard", "revision", "attempt", "digest")
    tree = ttk.Treeview(list_frame, columns=columns, show="headings")
    for col, width in zip(columns, (120, 90, 80, 420)):
        tree.heading(col, text=col)
        tree.column(col, width=width, anchor="w")
    tree.pack(fill="both", expand=True)

    text = tk.Text(detail_frame, wrap="word", font=("monospace", 10))
    text.pack(fill="both", expand=True)

    conn = sqlite3.connect(f"file:{ledger.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT shard_id,revision,attempt,summary_digest,outcome_canonical FROM current_view ORDER BY shard_id").fetchall()
    details: dict[str, str] = {}
    for row in rows:
        item_id = tree.insert("", "end", values=(row["shard_id"], row["revision"], row["attempt"], row["summary_digest"]))
        details[item_id] = row["outcome_canonical"]

    snapshot_text = ""
    if args.snapshot and Path(args.snapshot).exists():
        snapshot_text = Path(args.snapshot).read_text(encoding="utf-8")

    def show_detail(_event=None) -> None:
        selection = tree.selection()
        text.delete("1.0", "end")
        if selection:
            text.insert("1.0", details.get(selection[0], "{}"))
        elif snapshot_text:
            text.insert("1.0", snapshot_text)

    tree.bind("<<TreeviewSelect>>", show_detail)
    if snapshot_text:
        text.insert("1.0", snapshot_text[:200_000])

    status = ttk.Label(root, text=f"{len(rows)} committed current-view shards", anchor="w", padding=6)
    status.pack(fill="x")
    if args.smoke:
        root.after(100, root.destroy)
    root.mainloop()
    conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
