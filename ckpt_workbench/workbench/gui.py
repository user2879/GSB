"""Tkinter diagnostic workbench with a stable text-rendering fallback.

On a standard local Windows Python install, tkinter is available and the GUI
window launches. In headless environments without a Tk shared library the same
data is rendered to text via :func:`render_text`, keeping the demo repeatable.
"""

from .recovery import list_unknown_shards

COLUMNS = ("checkpoint", "parent", "version", "step", "shards", "mode")


def collect_view(ledger, snapshot_dirs=None):
    rows = []
    for cid in ledger.list_checkpoint_ids():
        entry = ledger.get_checkpoint(cid)
        rows.append((
            entry["checkpoint_id"],
            entry["parent_id"] or "-",
            entry["ledger_version"],
            entry["step"],
            entry["shard_count"],
            entry["partition_mode"],
        ))
    affected = []
    for record in ledger.list_changes():
        affected.append((record["seq"], record["checkpoint_id"], record["ledger_version"]))
    unknown = list_unknown_shards(ledger)
    snapshots = sorted(snapshot_dirs or [])
    return {
        "checkpoints": rows,
        "change_trace": affected,
        "unknown_shards": unknown,
        "snapshots": snapshots,
    }


def render_text(view):
    lines = ["== Checkpoint Dependency Chain =="]
    lines.append("  " + " | ".join(COLUMNS))
    for row in view["checkpoints"]:
        lines.append("  " + " | ".join(str(cell) for cell in row))

    lines.append("")
    lines.append("== Change Trace (stable seq order) ==")
    for seq, cid, version in view["change_trace"]:
        lines.append(f"  seq={seq} checkpoint={cid} ledger_version={version}")

    lines.append("")
    lines.append("== Affected / Unknown Shards ==")
    if view["unknown_shards"]:
        for item in view["unknown_shards"]:
            lines.append(f"  UNKNOWN {item['shard_id']} ({item['checkpoint_id']})")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("== Snapshots ==")
    for path in view["snapshots"]:
        lines.append(f"  {path}")
    if not view["snapshots"]:
        lines.append("  (none)")
    return "\n".join(lines)


def launch_gui(ledger, snapshot_dirs=None):
    view = collect_view(ledger, snapshot_dirs)
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        print(render_text(view))
        print(f"\n[tkinter unavailable in this environment: {exc!r}; text view used]")
        return "text"

    root = tk.Tk()
    root.title("Checkpoint & Recovery Diagnostic Workbench")

    tree = ttk.Treeview(root, columns=COLUMNS, show="headings", height=10)
    for column in COLUMNS:
        tree.heading(column, text=column)
        tree.column(column, width=120, anchor="w")
    for row in view["checkpoints"]:
        tree.insert("", "end", values=row)
    tree.pack(fill="both", expand=True, padx=8, pady=8)

    trace_box = tk.Text(root, height=10, wrap="none")
    trace_box.pack(fill="both", expand=True, padx=8)
    trace_box.insert("end", render_text(view))
    trace_box.configure(state="disabled")

    root.mainloop()
    return "gui"
