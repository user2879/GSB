"""Tkinter workbench. Imported lazily so headless tests remain dependency-free."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import StateError, TrackerError
from .service import ExperimentService


def _format_metrics(metrics: dict[str, float]) -> str:
    return f"loss={metrics['loss']:.6f}  accuracy={metrics['accuracy']:.6f}"


class TrackerWindow:
    def __init__(self, workspace: Path, service: ExperimentService | None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.workspace = workspace
        self.service = service
        self.root = tk.Tk()
        self.root.title("Local Experiment Tracker")
        self.root.geometry("1240x800")
        self.root.minsize(980, 680)
        self.run_items: dict[str, str] = {}
        self.snapshot_items: dict[str, int] = {}

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text=str(workspace), font=("TkDefaultFont", 9, "bold")).pack(
            side="left", padx=(0, 16)
        )
        ttk.Button(toolbar, text="Init from JSON", command=self._initialize).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="Save Snapshot", command=self._snapshot).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="Restore Snapshot", command=self._restore).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="Replay Snapshot", command=self._replay).pack(
            side="left", padx=3
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(12, 2)).pack(
            fill="x"
        )

        main_pane = ttk.PanedWindow(self.root, orient="vertical")
        main_pane.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        upper = ttk.PanedWindow(main_pane, orient="horizontal")
        main_pane.add(upper, weight=4)

        left = ttk.Frame(upper, padding=4)
        upper.add(left, weight=2)
        ttk.Label(left, text="Runs", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        run_columns = (
            "run_id",
            "rev",
            "format",
            "source",
            "aggregation",
            "parent",
            "final",
        )
        self.run_tree = ttk.Treeview(left, columns=run_columns, show="headings", height=14)
        headings = {
            "run_id": "Run",
            "rev": "Rev",
            "format": "Format",
            "source": "Source",
            "aggregation": "Aggregation",
            "parent": "Parent",
            "final": "Final metrics",
        }
        widths = {
            "run_id": 125,
            "rev": 45,
            "format": 62,
            "source": 70,
            "aggregation": 110,
            "parent": 90,
            "final": 255,
        }
        for column in run_columns:
            self.run_tree.heading(column, text=headings[column])
            self.run_tree.column(column, width=widths[column], anchor="w", stretch=True)
        run_scroll = ttk.Scrollbar(left, orient="vertical", command=self.run_tree.yview)
        self.run_tree.configure(yscrollcommand=run_scroll.set)
        self.run_tree.pack(side="left", fill="both", expand=True)
        run_scroll.pack(side="right", fill="y")
        self.run_tree.bind("<<TreeviewSelect>>", self._select_run)

        right = ttk.Frame(upper, padding=4)
        upper.add(right, weight=3)
        ttk.Label(right, text="Metrics and replacement history", font=("TkDefaultFont", 11, "bold")).pack(
            anchor="w"
        )
        self.canvas = tk.Canvas(right, background="#fbfbfb", highlightthickness=1, height=330)
        self.canvas.pack(fill="both", expand=True, pady=(3, 6))
        self.detail_text = tk.Text(right, height=12, wrap="word", font=("Consolas", 9))
        self.detail_text.pack(fill="both", expand=False)
        self.detail_text.configure(state="disabled")

        bottom = ttk.Frame(main_pane, padding=4)
        main_pane.add(bottom, weight=2)
        ttk.Label(bottom, text="Snapshots", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        snapshot_columns = (
            "id",
            "label",
            "runs",
            "formats",
            "sources",
            "aggregations",
            "checkpoint",
        )
        self.snapshot_tree = ttk.Treeview(
            bottom, columns=snapshot_columns, show="headings", height=8
        )
        snapshot_headings = {
            "id": "ID",
            "label": "Label",
            "runs": "Runs",
            "formats": "Formats",
            "sources": "Sources",
            "aggregations": "Aggregation versions",
            "checkpoint": "Checkpoint path / position",
        }
        snapshot_widths = {
            "id": 45,
            "label": 190,
            "runs": 55,
            "formats": 90,
            "sources": 100,
            "aggregations": 150,
            "checkpoint": 360,
        }
        for column in snapshot_columns:
            self.snapshot_tree.heading(column, text=snapshot_headings[column])
            self.snapshot_tree.column(
                column, width=snapshot_widths[column], anchor="w", stretch=True
            )
        snapshot_scroll = ttk.Scrollbar(
            bottom, orient="vertical", command=self.snapshot_tree.yview
        )
        self.snapshot_tree.configure(yscrollcommand=snapshot_scroll.set)
        self.snapshot_tree.pack(side="left", fill="both", expand=True)
        snapshot_scroll.pack(side="right", fill="y")

        self.refresh()

    def run(self) -> None:
        self.root.mainloop()

    def refresh(self) -> None:
        for item in self.run_tree.get_children():
            self.run_tree.delete(item)
        for item in self.snapshot_tree.get_children():
            self.snapshot_tree.delete(item)
        self.run_items.clear()
        self.snapshot_items.clear()
        self._clear_canvas()
        self._set_detail("Initialize a workspace or select a run.")
        if self.service is None:
            self.status_var.set("No record library loaded. Use Init from JSON.")
            return
        runs = self.service.list_runs()
        for run in runs:
            item = self.run_tree.insert(
                "",
                "end",
                values=(
                    run["run_id"],
                    run["run_version"],
                    f"v{run['format_version']}",
                    f"v{run['source_format_version']}→v{run['format_version']}"
                    if run["upgraded"]
                    else f"v{run['source_format_version']}",
                    f"{run['aggregation']['name']} ({run['aggregation']['version']})",
                    run["parent_run_id"] or "—",
                    _format_metrics(run["final_metrics"]),
                ),
            )
            self.run_items[item] = run["run_id"]
        snapshots = self.service.list_snapshots()
        for snapshot in snapshots:
            item = self.snapshot_tree.insert(
                "",
                "end",
                values=(
                    snapshot["snapshot_id"],
                    snapshot["label"],
                    snapshot["run_count"],
                    ",".join(f"v{value}" for value in snapshot["format_versions"]),
                    ",".join(f"v{value}" for value in snapshot["source_format_versions"]),
                    ",".join(snapshot["aggregation_versions"]),
                    f"{snapshot['checkpoint_path']} @ {snapshot['checkpoint_position']}",
                ),
            )
            self.snapshot_items[item] = snapshot["snapshot_id"]
        self.status_var.set(
            f"{len(runs)} runs · formats {self.service.library.format_versions()} · "
            f"sources {self.service.library.source_format_versions()} · {len(snapshots)} snapshots"
        )
        if runs:
            first = self.run_tree.get_children()[0]
            self.run_tree.selection_set(first)
            self.run_tree.focus(first)

    def _selected_run_id(self) -> str | None:
        selected = self.run_tree.selection()
        if not selected:
            return None
        return self.run_items.get(selected[0])

    def _selected_snapshot_id(self) -> int | None:
        selected = self.snapshot_tree.selection()
        if not selected:
            return None
        return self.snapshot_items.get(selected[0])

    def _select_run(self, _event: Any = None) -> None:
        if self.service is None:
            return
        run_id = self._selected_run_id()
        if not run_id:
            return
        detail = self.service.run_detail(run_id)
        current = detail["run"]
        lines = [
            f"Run: {current['run_id']}    revision: {current['run_version']}",
            f"Record format: v{current['format_version']}    source: v{current['source_format_version']}    upgraded: {current['upgraded']}",
            f"Aggregation: {current['aggregation']['name']} / {current['aggregation']['version']}",
            f"Parent: {current['parent_run_id'] or '—'}",
            f"Final metrics: {_format_metrics(current['final_metrics'])}",
            f"Metric digest: {current['metric_digest']}",
            "",
            "History (read-only):",
        ]
        if not detail["history"]:
            lines.append("  none")
        for record in detail["history"]:
            lines.append(
                f"  v{record['run_version']} — format v{record['format_version']} — "
                f"{_format_metrics(record['final_metrics'])}"
            )
        if "replacement" in detail:
            replacement = detail["replacement"]
            lines.extend(["", "Last whole-record replacement:"])
            for change in replacement["changed_fields"]:
                lines.append(
                    f"  {change['field']}: {change['before']!r} → {change['after']!r}"
                )
            metrics = replacement["metric_changes"]
            lines.append(
                f"  first changed step: {metrics['first_changed_step']}    "
                f"changed steps: {metrics['changed_step_count']}"
            )
            lines.append(
                f"  final loss Δ {metrics['final_loss']['delta']:+.9f}    "
                f"accuracy Δ {metrics['final_accuracy']['delta']:+.9f}"
            )
        self._set_detail("\n".join(lines))
        self._draw_curve(detail["step_metrics"])

    def _set_detail(self, value: str) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", value)
        self.detail_text.configure(state="disabled")

    def _clear_canvas(self) -> None:
        self.canvas.delete("all")

    def _draw_curve(self, points: list[dict[str, Any]]) -> None:
        self._clear_canvas()
        width = max(self.canvas.winfo_width(), 420)
        height = max(self.canvas.winfo_height(), 260)
        left, top, right, bottom = 54, 24, width - 24, height - 42
        self.canvas.create_rectangle(left, top, right, bottom, outline="#c8c8c8")
        if len(points) < 2:
            self.canvas.create_text(width / 2, height / 2, text="No metric points")
            return
        values = [point["loss"] for point in points] + [point["accuracy"] for point in points]
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            maximum = minimum + 1.0
        padding = (maximum - minimum) * 0.08
        minimum -= padding
        maximum += padding
        span = maximum - minimum

        def coord(step: int, value: float) -> tuple[float, float]:
            x = left + (step - 1) * (right - left) / (len(points) - 1)
            y = bottom - (value - minimum) * (bottom - top) / span
            return x, y

        for tick in range(5):
            y = top + tick * (bottom - top) / 4
            value = maximum - tick * span / 4
            self.canvas.create_line(left, y, right, y, fill="#ececec")
            self.canvas.create_text(left - 8, y, text=f"{value:.3f}", anchor="e", font=("TkDefaultFont", 8))
        step_ticks = sorted({1, len(points), max(1, len(points) // 2)})
        for step in step_ticks:
            x, _ = coord(step, minimum)
            self.canvas.create_text(x, bottom + 14, text=str(step), font=("TkDefaultFont", 8))
        self.canvas.create_text((left + right) / 2, height - 12, text="training step")

        for field, color, label in (("loss", "#c0392b", "loss"), ("accuracy", "#1f6fb2", "accuracy")):
            coords = []
            for point in points:
                coords.extend(coord(point["step"], point[field]))
            self.canvas.create_line(*coords, fill=color, width=2, smooth=False)
            self.canvas.create_text(
                right - 82,
                top + (10 if field == "loss" else 28),
                text=label,
                fill=color,
                anchor="w",
                font=("TkDefaultFont", 9, "bold"),
            )

    def _initialize(self) -> None:
        from tkinter import filedialog, messagebox

        source = filedialog.askopenfilename(
            title="Select experiment declaration JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not source:
            return
        force = False
        if (self.workspace / "records.json").exists():
            force = messagebox.askyesno(
                "Replace session",
                "A record library already exists. Replace the app-owned session state?",
            )
            if not force:
                return
        try:
            self.service = ExperimentService.initialize(
                self.workspace, Path(source), force=force
            )
            self.refresh()
            self.status_var.set("Initialized and auto-upgraded from local JSON")
        except TrackerError as exc:
            messagebox.showerror(exc.code, _json_error(exc))

    def _snapshot(self) -> None:
        from tkinter import messagebox, simpledialog

        if self.service is None:
            messagebox.showinfo("Snapshot", "Initialize a workspace first.")
            return
        label = simpledialog.askstring("Save snapshot", "Snapshot label:", parent=self.root)
        if not label:
            return
        try:
            result = self.service.snapshot(label)
            self.refresh()
            messagebox.showinfo("Snapshot saved", _pretty(result))
        except TrackerError as exc:
            messagebox.showerror(exc.code, _json_error(exc))

    def _restore(self) -> None:
        from tkinter import messagebox

        if self.service is None:
            return
        snapshot_id = self._selected_snapshot_id()
        if snapshot_id is None:
            messagebox.showinfo("Restore", "Select a snapshot first.")
            return
        if not messagebox.askyesno(
            "Restore snapshot",
            "This will replace the current record library and checkpoint as a whole. Continue?",
        ):
            return
        try:
            result = self.service.restore_snapshot(snapshot_id, replay=True)
            self.refresh()
            messagebox.showinfo("Snapshot restored and replayed", _pretty(result))
        except TrackerError as exc:
            messagebox.showerror(exc.code, _json_error(exc))

    def _replay(self) -> None:
        from tkinter import messagebox

        if self.service is None:
            return
        snapshot_id = self._selected_snapshot_id()
        if snapshot_id is None:
            messagebox.showinfo("Replay", "Select a snapshot first.")
            return
        try:
            result = self.service.replay_snapshot(snapshot_id)
            messagebox.showinfo("Replay passed", _pretty(result))
        except TrackerError as exc:
            messagebox.showerror(exc.code, _json_error(exc))


def _pretty(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _json_error(error: TrackerError) -> str:
    return _pretty(error.to_dict())


def launch(workspace: Path) -> None:
    workspace = workspace.resolve()
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        raise StateError(
            "tkinter is not available in this Python installation",
            hint="install the official Windows Python distribution with Tcl/Tk",
        ) from exc
    service: ExperimentService | None
    try:
        service = ExperimentService.open(workspace)
    except StateError:
        service = None
    try:
        window = TrackerWindow(workspace, service)
    except Exception as exc:
        raise StateError("cannot start the Tkinter workbench", error=str(exc)) from exc
    window.run()

