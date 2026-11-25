from __future__ import annotations

import argparse
from pathlib import Path

from .dashboard import build_dashboard_model, render_text


def write_text_snapshot(spec_dir: str | Path, ledger: str | Path, output: str | Path) -> Path:
    model = build_dashboard_model(spec_dir, ledger)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_text(model), encoding="utf-8")
    return path


def run_tk(spec_dir: str | Path, ledger: str | Path, *, smoke_ms: int | None = None) -> int:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Edge CDN Ledger Diagnostics")
    root.geometry("1180x760")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    title = ttk.Label(top, text="Edge cache ledger", font=("Sans", 14, "bold"))
    title.pack(side="left")
    meta_var = tk.StringVar(value="loading...")
    ttk.Label(top, textvariable=meta_var).pack(side="left", padx=16)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    water_frame = ttk.Frame(notebook)
    shard_frame = ttk.Frame(notebook)
    receipt_frame = ttk.Frame(notebook)
    notebook.add(water_frame, text="Lease watermarks")
    notebook.add(shard_frame, text="Shard conclusions")
    notebook.add(receipt_frame, text="Receipt decisions")

    water_tree = _make_tree(
        water_frame,
        ("node_id", "quota", "active", "capacity", "effective_cap", "available", "watermark"),
    )
    shard_tree = _make_tree(
        shard_frame,
        ("shard_id", "node_id", "status", "unknown", "digest", "content_digest", "format"),
    )
    receipt_tree = _make_tree(
        receipt_frame,
        ("id", "quota", "shard", "node", "format", "line", "adopted", "current", "stale", "reason", "filled"),
    )

    status_var = tk.StringVar(value="Read-only refresh; ledger is never written by UI.")
    ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def refresh() -> None:
        model = build_dashboard_model(spec_dir, ledger)
        meta_var.set(
            f"ledger v{model.ledger_format_version} | content {model.content_digest} | "
            f"quota v{model.spec_quota_version} | unknown {len(model.unknown_shards)}"
        )
        _replace_rows(
            water_tree,
            [
                (
                    row.node_id,
                    row.quota_version,
                    row.active_leases,
                    row.capacity_slots,
                    row.effective_cap,
                    row.available_slots,
                    row.watermark,
                )
                for row in model.node_water
            ],
        )
        _replace_rows(
            shard_tree,
            [
                (
                    row.shard_id,
                    row.node_id,
                    row.status,
                    "UNKNOWN/RERUN" if row.unknown else "ok",
                    row.digest,
                    row.content_digest,
                    row.source_format,
                )
                for row in model.shards
            ],
        )
        _replace_rows(
            receipt_tree,
            [
                (
                    row.receipt_id,
                    row.quota_version,
                    row.shard_id,
                    row.node_id,
                    row.source_format,
                    row.source_line,
                    "yes" if row.adopted else "no",
                    "yes" if row.current else "no",
                    "yes" if row.stale else "no",
                    row.reason,
                    row.filled_fields,
                )
                for row in model.receipts
            ],
        )
        status_var.set(f"Refreshed {len(model.node_water)} nodes, {len(model.shards)} shard rows, {len(model.receipts)} receipts.")

    buttons = ttk.Frame(root, padding=(8, 0))
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Refresh (read-only)", command=refresh).pack(side="left")
    ttk.Button(buttons, text="Close", command=root.destroy).pack(side="left", padx=8)
    refresh()

    if smoke_ms is not None:
        root.after(smoke_ms, root.destroy)
    root.mainloop()
    return 0


def _make_tree(parent: object, columns: tuple[str, ...]) -> object:
    from tkinter import ttk

    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150 if col in {"reason", "filled"} else 110, anchor="w")
    yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=yscroll.set)
    tree.pack(side="left", fill="both", expand=True)
    yscroll.pack(side="right", fill="y")
    return tree


def _replace_rows(tree: object, rows: list[tuple[object, ...]]) -> None:
    tree.delete(*tree.get_children())
    for row in rows:
        tree.insert("", "end", values=row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tkinter diagnostic UI")
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--text-snapshot")
    parser.add_argument("--smoke-ms", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.text_snapshot:
        path = write_text_snapshot(args.spec_dir, args.ledger, args.text_snapshot)
        print(f"text snapshot written: {path}")
        return 0
    return run_tk(args.spec_dir, args.ledger, smoke_ms=args.smoke_ms)


if __name__ == "__main__":
    raise SystemExit(main())
