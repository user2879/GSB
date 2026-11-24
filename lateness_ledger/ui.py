"""Tkinter diagnostics interface with stable, read-only rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import UiError
from .ui_model import load_dashboard, pretty_partition


def launch_dashboard(ledger_path: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise UiError(
            "tkinter is not available in this Python runtime",
            details={
                "cause": str(exc),
                "windows_fix": "use the official Windows Python installer with Tcl/Tk",
            },
        ) from exc

    try:
        root = tk.Tk()
    except Exception as exc:  # noqa: BLE001 - Tcl display availability is environment-specific
        raise UiError(
            "Tk could not create a display context",
            details={"cause": str(exc), "ledger": str(ledger_path)},
        ) from exc
    root.title("Late Event Displacement & Watermark Ledger")
    root.geometry("1320x820")

    state: dict[str, Any] = {"data": load_dashboard(ledger_path)}
    header = ttk.Frame(root, padding=(10, 8))
    header.pack(fill="x")
    title_var = tk.StringVar()
    ttk.Label(header, textvariable=title_var, font=("TkDefaultFont", 12, "bold")).pack(
        side="left"
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    overview_frame = ttk.Frame(notebook, padding=12)
    watermark_frame = ttk.Frame(notebook, padding=8)
    windows_frame = ttk.Frame(notebook, padding=8)
    late_frame = ttk.Frame(notebook, padding=8)
    displacement_frame = ttk.Frame(notebook, padding=8)
    error_frame = ttk.Frame(notebook, padding=8)
    snapshot_frame = ttk.Frame(notebook, padding=8)
    notebook.add(overview_frame, text="Overview")
    notebook.add(watermark_frame, text="Partition Watermarks")
    notebook.add(windows_frame, text="Finalized Windows")
    notebook.add(late_frame, text="Late Decisions")
    notebook.add(displacement_frame, text="Displacement Chains")
    notebook.add(error_frame, text="Fixed-Point Errors")
    notebook.add(snapshot_frame, text="Snapshots")

    overview_text = tk.Text(overview_frame, wrap="word", height=20)
    overview_text.pack(fill="both", expand=True)
    overview_text.configure(state="disabled")

    watermark_tree = ttk.Treeview(
        watermark_frame,
        columns=("event_seq", "phase", "partition_watermark", "global_watermark"),
        show="headings",
    )
    for column, label in (
        ("event_seq", "Event Seq"),
        ("phase", "Phase"),
        ("partition_watermark", "Partition WM"),
        ("global_watermark", "Global WM"),
    ):
        watermark_tree.heading(column, text=label)
        watermark_tree.column(column, width=120, anchor="e" if "watermark" in column else "w")
    watermark_tree.pack(side="left", fill="both", expand=True)
    watermark_canvas = tk.Canvas(watermark_frame, background="#101820", width=520)
    watermark_canvas.pack(side="right", fill="both", expand=True, padx=(8, 0))

    windows_tree = ttk.Treeview(
        windows_frame,
        columns=("order", "rule", "partition", "boundary", "trigger", "version", "count", "sum"),
        show="headings",
    )
    for column, label, width in (
        ("order", "Order", 60),
        ("rule", "Rule", 100),
        ("partition", "Partition", 170),
        ("boundary", "Start..End", 150),
        ("trigger", "Trigger Seq", 90),
        ("version", "Version", 70),
        ("count", "Count", 70),
        ("sum", "Sum Fixed", 150),
    ):
        windows_tree.heading(column, text=label)
        windows_tree.column(column, width=width, anchor="w")
    windows_tree.pack(fill="both", expand=True)

    late_tree = ttk.Treeview(
        late_frame,
        columns=("decision", "window", "event", "seq", "remaining_count", "remaining_abs", "rule"),
        show="headings",
    )
    for column, label, width in (
        ("decision", "Decision", 90),
        ("window", "Window", 330),
        ("event", "Event", 180),
        ("seq", "Seq", 70),
        ("remaining_count", "Remaining Count", 120),
        ("remaining_abs", "Remaining Abs", 130),
        ("rule", "Rule", 260),
    ):
        late_tree.heading(column, text=label)
        late_tree.column(column, width=width, anchor="w")
    late_tree.pack(fill="both", expand=True)

    displacement_tree = ttk.Treeview(
        displacement_frame,
        columns=("root", "reason", "path", "changed", "versions", "basis"),
        show="headings",
    )
    for column, label, width in (
        ("root", "Root", 300),
        ("reason", "Reason", 240),
        ("path", "Stable Path", 430),
        ("changed", "Changed", 80),
        ("versions", "Version", 100),
        ("basis", "Basis", 330),
    ):
        displacement_tree.heading(column, text=label)
        displacement_tree.column(column, width=width, anchor="w")
    displacement_tree.pack(fill="both", expand=True)

    error_tree = ttk.Treeview(
        error_frame,
        columns=("window", "boundary", "quant", "late", "boundary_err", "total", "mean", "dominant"),
        show="headings",
    )
    for column, label, width in (
        ("window", "Window", 340),
        ("boundary", "Start..End", 140),
        ("quant", "Quantization", 100),
        ("late", "Late", 100),
        ("boundary_err", "Boundary", 100),
        ("total", "Agg Bound", 100),
        ("mean", "Mean Bound", 100),
        ("dominant", "Dominant", 140),
    ):
        error_tree.heading(column, text=label)
        error_tree.column(column, width=width, anchor="w")
    error_tree.pack(fill="both", expand=True)

    snapshot_tree = ttk.Treeview(
        snapshot_frame,
        columns=("snapshot_id", "created", "path", "digest"),
        show="headings",
    )
    for column, label, width in (
        ("snapshot_id", "Snapshot ID", 240),
        ("created", "Created", 190),
        ("path", "Path", 620),
        ("digest", "Content Digest", 420),
    ):
        snapshot_tree.heading(column, text=label)
        snapshot_tree.column(column, width=width, anchor="w")
    snapshot_tree.pack(fill="both", expand=True)

    def clear_tree(tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def refresh() -> None:
        data = load_dashboard(ledger_path)
        state["data"] = data
        meta = data["meta"]
        title_var.set(
            f"{meta.get('session_id', '<unknown>')}  |  strategy={meta.get('strategy', {}).get('kind', '?')}  "
            f"| scale={meta.get('scale', '?')}  | status={meta.get('run_status', '?')}"
        )
        overview_text.configure(state="normal")
        overview_text.delete("1.0", "end")
        overview_lines = [
            f"Ledger: {data['ledger']}",
            f"Rules version: {meta.get('rules_version')}",
            f"Spec digest: {meta.get('spec_digest')}",
            f"Event cursor: {meta.get('event_cursor')} / generated {meta.get('events_generated')}",
            f"Windows: {len(data['window_rows'])}",
            f"Late decisions: {len(data['late_rows'])}",
            f"Displacement runs: {len(data['displacements'])}",
            f"Snapshots: {len(data['snapshots'])}",
            "",
            "Rendering is read-only. Refresh reloads the SQLite ledger and never writes to it.",
        ]
        overview_text.insert("1.0", "\n".join(overview_lines))
        overview_text.configure(state="disabled")

        clear_tree(watermark_tree)
        for partition in data["partition_rows"]:
            for point in partition["points"]:
                watermark_tree.insert(
                    "",
                    "end",
                    values=(
                        point["event_seq"],
                        point["phase"],
                        point["partition_watermark"],
                        point["global_watermark"],
                    ),
                )
        watermark_canvas.delete("all")
        all_points = [
            point
            for partition in data["partition_rows"]
            for point in partition["points"]
        ]
        max_global = max((int(p["global_watermark"]) for p in all_points), default=1)
        max_seq = max((int(p["event_seq"]) for p in all_points), default=1)
        colors = ("#57c7ff", "#ffb74d", "#7ee787", "#f778ba", "#c792ea")
        canvas_width = max(watermark_canvas.winfo_width(), 520)
        canvas_height = max(watermark_canvas.winfo_height(), 500)
        watermark_canvas.create_text(10, 10, anchor="nw", fill="#d9e2ec", text="Global watermark trajectory")
        for index, partition in enumerate(data["partition_rows"]):
            rows = [p for p in partition["points"] if p["phase"] != "init"]
            previous = None
            color = colors[index % len(colors)]
            for point in rows:
                x = 24 + int((int(point["event_seq"]) + 1) / max(1, max_seq + 1) * (canvas_width - 40))
                y = canvas_height - 20 - int(int(point["global_watermark"]) / max(1, max_global) * (canvas_height - 50))
                if previous is not None:
                    watermark_canvas.create_line(*previous, x, y, fill=color, width=1)
                previous = (x, y)
            watermark_canvas.create_text(
                canvas_width - 10,
                28 + index * 18,
                anchor="ne",
                fill=color,
                text=pretty_partition(partition["partition"]),
            )

        clear_tree(windows_tree)
        for row in data["window_rows"]:
            windows_tree.insert(
                "",
                "end",
                values=(
                    row["finalization_order"] if row["finalization_order"] is not None else "UNKNOWN",
                    row["rule_id"],
                    pretty_partition(row["partition"]),
                    f"{row['start_ms']}..{row['end_ms']}",
                    row["trigger_seq"] if row["trigger_seq"] is not None else "-",
                    row["version"],
                    row["count"] if row["count"] is not None else "-",
                    row["sum_fixed"] if row["sum_fixed"] is not None else "-",
                ),
            )

        clear_tree(late_tree)
        for row in data["late_rows"]:
            late_tree.insert(
                "",
                "end",
                values=(
                    row["decision"],
                    row["window_id"],
                    row["event_id"],
                    row["event_seq"],
                    row["remaining_event_count"],
                    row["remaining_absolute_measure_fixed"],
                    row["rule"],
                ),
            )

        clear_tree(displacement_tree)
        for run in data["displacements"]:
            for step in run["affected"]:
                displacement_tree.insert(
                    "",
                    "end",
                    values=(
                        run["root_window_id"],
                        run["reason"],
                        " -> ".join(step["path"]),
                        "yes" if step["changed"] else "no",
                        f"{step['version_before']}->{step['version_after']}",
                        "; ".join(item.get("reason", "") for item in step["basis"]),
                    ),
                )

        clear_tree(error_tree)
        for row in data["error_rows"]:
            segments = row.get("segments_fixed", {})
            error_tree.insert(
                "",
                "end",
                values=(
                    row["window_id"],
                    f"{row['start_ms']}..{row['end_ms']}",
                    segments.get("quantization_truncation", "-"),
                    segments.get("late_admission", "-"),
                    segments.get("boundary_attribution", "-"),
                    row.get("aggregation_error_bound_fixed", "-"),
                    row.get("mean_error_bound_fixed", "-"),
                    f"{row.get('dominant_segment')} ({row.get('dominant_share')})",
                ),
            )

        clear_tree(snapshot_tree)
        for snapshot in data["snapshots"]:
            snapshot_tree.insert(
                "",
                "end",
                values=(
                    snapshot["snapshot_id"],
                    snapshot["created_at"],
                    snapshot["path"],
                    snapshot["content_digest"],
                ),
            )

    refresh_button = ttk.Button(header, text="Refresh (read-only)", command=refresh)
    refresh_button.pack(side="right")
    refresh()
    root.mainloop()
