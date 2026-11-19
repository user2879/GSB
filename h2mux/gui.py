"""Tkinter dashboard for the offline HTTP/2 state laboratory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:  # pragma: no cover - exercised on headless Linux
    tk = None
    ttk = None
    messagebox = None
    filedialog = None

from .declaration import load_declaration, load_script
from .replay import replay_from_snapshot
from .session import H2Session
from .snapshot import SnapshotStore, restore_snapshot
from .viewmodel import build_viewmodel


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


if tk is not None:

    class H2MuxApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("HTTP/2 Multiplexing State Laboratory")
            self.geometry("1280x820")
            self.minsize(980, 640)
            root = _project_root()
            self.declaration = load_declaration(root / "samples" / "session.json")
            self.script = load_script(root / "samples" / "frames.json", self.declaration)
            self.snapshot_store = SnapshotStore(root / "artifacts" / "gui_snapshots")
            self.session: H2Session | None = None
            self.view: dict[str, Any] = {}
            self._snapshot_paths: dict[str, str] = {}
            self._build_controls()
            self._build_tabs()
            self._new_session("weighted_round_robin")
            self.refresh()

        def _build_controls(self) -> None:
            bar = ttk.Frame(self, padding=(8, 6))
            bar.pack(side=tk.TOP, fill=tk.X)
            ttk.Label(bar, text="Strategy").pack(side=tk.LEFT)
            self.strategy_var = tk.StringVar(value="weighted_round_robin")
            self.strategy_combo = ttk.Combobox(
                bar,
                textvariable=self.strategy_var,
                values=("weighted_round_robin", "window_priority"),
                width=26,
                state="readonly",
            )
            self.strategy_combo.pack(side=tk.LEFT, padx=(6, 14))
            ttk.Button(bar, text="New session", command=self._new_session_from_combo).pack(side=tk.LEFT, padx=3)
            ttk.Button(bar, text="Load JSON", command=self._load_json).pack(side=tk.LEFT, padx=3)
            ttk.Button(bar, text="Run stage", command=self._run_stage).pack(side=tk.LEFT, padx=3)
            ttk.Button(bar, text="Continue", command=self._continue).pack(side=tk.LEFT, padx=3)
            ttk.Button(bar, text="Switch strategy", command=self._switch_strategy).pack(side=tk.LEFT, padx=3)
            ttk.Button(bar, text="Save snapshot", command=self._save_snapshot).pack(side=tk.LEFT, padx=3)
            ttk.Button(bar, text="Restore + replay", command=self._restore_replay).pack(side=tk.LEFT, padx=3)
            self.status_var = tk.StringVar(value="Ready")
            ttk.Label(bar, textvariable=self.status_var).pack(side=tk.RIGHT)

        def _build_tabs(self) -> None:
            book = ttk.Notebook(self)
            book.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
            self.book = book

            frame_tab = ttk.Frame(book)
            book.add(frame_tab, text="Frame timeline")
            frame_columns = (
                "seq",
                "frame",
                "dir",
                "type",
                "stream",
                "bytes",
                "flow",
                "conn_after",
                "stream_after",
                "state",
                "code",
            )
            self.frame_tree = self._tree(frame_tab, frame_columns)
            for column, text, width in (
                ("seq", "#", 50),
                ("frame", "Frame", 65),
                ("dir", "Dir", 55),
                ("type", "Type", 105),
                ("stream", "Stream", 65),
                ("bytes", "Payload", 75),
                ("flow", "Flow", 65),
                ("conn_after", "Conn S->", 85),
                ("stream_after", "Stream S->", 90),
                ("state", "State", 205),
                ("code", "Code", 75),
            ):
                self.frame_tree.heading(column, text=text)
                self.frame_tree.column(column, width=width, stretch=True)

            window_tab = ttk.Frame(book)
            book.add(window_tab, text="Window curves")
            self.window_canvas = tk.Canvas(window_tab, background="#fbfbf8", highlightthickness=0)
            self.window_canvas.pack(fill=tk.BOTH, expand=True)
            self.window_canvas.bind("<Configure>", lambda _event: self._draw_windows())

            state_tab = ttk.Frame(book)
            book.add(state_tab, text="State transitions")
            self.state_tree = self._tree(state_tab, ("transition", "stream", "frame", "clock", "trigger"))
            for column, text, width in (
                ("transition", "Transition", 260),
                ("stream", "Stream", 70),
                ("frame", "Frame", 70),
                ("clock", "Clock ms", 80),
                ("trigger", "Trigger", 340),
            ):
                self.state_tree.heading(column, text=text)
                self.state_tree.column(column, width=width, stretch=True)

            priority_tab = ttk.Frame(book)
            book.add(priority_tab, text="Priority & fairness")
            self.priority_tree = self._tree(
                priority_tab,
                ("tree", "parent", "weight", "share", "expected", "actual", "deviation"),
            )
            for column, text, width in (
                ("tree", "Stream tree", 140),
                ("parent", "Parent", 70),
                ("weight", "Weight", 70),
                ("share", "Share", 100),
                ("expected", "Expected bytes", 120),
                ("actual", "Actual bytes", 110),
                ("deviation", "Deviation", 90),
            ):
                self.priority_tree.heading(column, text=text)
                self.priority_tree.column(column, width=width, stretch=True)

            wait_tab = ttk.Frame(book)
            book.add(wait_tab, text="Empty waits")
            self.wait_tree = self._tree(
                wait_tab, ("kind", "id", "reason", "streams", "start", "end", "closed_by")
            )
            for column, text, width in (
                ("kind", "Kind", 110),
                ("id", "ID", 45),
                ("reason", "Reason", 260),
                ("streams", "Streams", 100),
                ("start", "Start ms", 80),
                ("end", "End ms", 80),
                ("closed_by", "Closed by", 190),
            ):
                self.wait_tree.heading(column, text=text)
                self.wait_tree.column(column, width=width, stretch=True)

            snapshot_tab = ttk.Frame(book)
            book.add(snapshot_tab, text="Snapshots")
            self.snapshot_tree = self._tree(
                snapshot_tab, ("name", "label", "version", "strategy", "index", "digest", "error")
            )
            for column, text, width in (
                ("name", "Name", 180),
                ("label", "Label", 150),
                ("version", "Decl ver", 75),
                ("strategy", "Strategy", 170),
                ("index", "Script idx", 80),
                ("digest", "Content digest", 240),
                ("error", "Error", 180),
            ):
                self.snapshot_tree.heading(column, text=text)
                self.snapshot_tree.column(column, width=width, stretch=True)

        def _tree(self, parent: Any, columns: tuple[str, ...]) -> ttk.Treeview:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.BOTH, expand=True)
            tree = ttk.Treeview(frame, columns=columns, show="headings")
            scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
            tree.grid(row=0, column=0, sticky="nsew")
            scroll_y.grid(row=0, column=1, sticky="ns")
            scroll_x.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            return tree

        def _new_session(self, strategy: str) -> None:
            self.session = H2Session(self.declaration, self.script, strategy=strategy)
            self.strategy_var.set(strategy)
            self.status_var.set(f"New {strategy} session")

        def _new_session_from_combo(self) -> None:
            self._new_session(self.strategy_var.get())
            self.refresh()

        def _load_json(self) -> None:
            if filedialog is None or messagebox is None:
                return
            declaration_path = filedialog.askopenfilename(
                title="Select session declaration JSON",
                initialdir=str(_project_root() / "samples"),
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            )
            if not declaration_path:
                return
            script_path = filedialog.askopenfilename(
                title="Select deterministic frame script JSON",
                initialdir=str(_project_root() / "samples"),
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            )
            if not script_path:
                return
            try:
                declaration = load_declaration(declaration_path)
                script = load_script(script_path, declaration)
            except Exception as exc:
                details = exc.to_dict() if hasattr(exc, "to_dict") else {"message": str(exc)}
                messagebox.showerror("Load failed", json.dumps(details, ensure_ascii=False, indent=2))
                return
            self.declaration = declaration
            self.script = script
            self._new_session(self.strategy_var.get())
            self.status_var.set(f"Loaded declaration v{declaration.declaration_version}: {script_path}")
            self.refresh()

        def _run_stage(self) -> None:
            if self.session is None:
                return
            self.session.run(stop_after_events=min(13, len(self.script.events)))
            self.status_var.set(f"Stopped at script event {self.session.script_index}")
            self.refresh()

        def _continue(self) -> None:
            if self.session is None:
                return
            self.session.run()
            self.status_var.set("Session complete" if self.session.completion_report()["complete"] else "Session stopped")
            self.refresh()

        def _switch_strategy(self) -> None:
            if self.session is None:
                return
            self.session.set_strategy(self.strategy_var.get())
            self.status_var.set(f"Strategy is now {self.session.strategy}")
            self.refresh()

        def _save_snapshot(self) -> None:
            if self.session is None:
                return
            path = self.snapshot_store.save(self.session, "gui-stage.json", label="GUI stage snapshot")
            self.status_var.set(f"Snapshot saved: {path}")
            self.refresh()

        def _restore_replay(self) -> None:
            if self.session is None:
                return
            selected = self.snapshot_tree.selection()
            selected_path = self._snapshot_paths.get(selected[0]) if selected else None
            path = Path(selected_path) if selected_path else self.snapshot_store._path("gui-stage.json")
            if not path.exists():
                if messagebox is not None:
                    messagebox.showinfo("Snapshot", "No snapshot is selected or saved yet.")
                return
            document = self.snapshot_store.load(path)
            target_strategy = self.strategy_var.get()
            restored = restore_snapshot(
                document.path,
                self.script,
                current_declaration=self.declaration,
                current_strategy=target_strategy,
            )
            self.session = restored.session
            report = replay_from_snapshot(
                str(document.path),
                self.script,
                current_declaration=self.declaration,
                current_strategy=target_strategy,
            )
            if report.status == "MATCH":
                self.status_var.set(
                    f"Replay MATCH: {report.compared_record_count} records from script index {report.snapshot_script_index}"
                )
                if restored.replacement_required and messagebox is not None:
                    messagebox.showinfo(
                        "Whole-state replacement",
                        "Snapshot state replaced the current session.\n"
                        + ", ".join(restored.replacement_reasons),
                    )
            else:
                self.status_var.set(f"Replay {report.status}: {report.reason}")
                if messagebox is not None:
                    messagebox.showinfo("Replay report", report.message)
            self.refresh()

        def refresh(self) -> None:
            if self.session is None:
                return
            self.strategy_var.set(self.session.strategy)
            view = build_viewmodel(self.session, self.snapshot_store)
            self.view = view
            self._fill_frames(view["frame_timeline"])
            self._fill_states(view["state_transitions"])
            self._fill_priority(view["priority_rows"], view["starvation"])
            self._fill_waits(view["waits"])
            self._fill_snapshots(view["snapshots"])
            self._draw_windows()

        def _clear_and_fill(self, tree: ttk.Treeview, rows: list[dict[str, Any]], values: tuple[str, ...]) -> None:
            children = tree.get_children("")
            if children:
                tree.delete(*children)
            for index, row in enumerate(rows):
                tree.insert(
                    "",
                    "end",
                    iid=f"row-{index}",
                    values=tuple("" if row.get(key) is None else row.get(key) for key in values),
                )

        def _fill_frames(self, rows: list[dict[str, Any]]) -> None:
            display = [
                {
                    **row,
                    "state": f"{row.get('state_before') or '-'} -> {row.get('state_after') or '-'}",
                }
                for row in rows
            ]
            self._clear_and_fill(
                self.frame_tree,
                display,
                ("seq", "frame_seq", "direction", "frame_type", "stream_id", "payload_bytes", "flow_bytes", "conn_send_window_after", "stream_send_window_after", "state", "code"),
            )

        def _fill_states(self, rows: list[dict[str, Any]]) -> None:
            self._clear_and_fill(
                self.state_tree,
                rows,
                ("transition", "stream_id", "frame_seq", "clock_ms", "trigger"),
            )

        def _fill_priority(self, rows: list[dict[str, Any]], starvation: list[dict[str, Any]]) -> None:
            display = [dict(row) for row in rows]
            for span in starvation:
                display.append(
                    {
                        "tree": f"  starvation {span['stream_id']}",
                        "parent": "",
                        "weight": "",
                        "share": "",
                        "expected": "",
                        "actual": "",
                        "deviation": f"{span['start_round']}..{span['end_round']}",
                    }
                )
            self._clear_and_fill(
                self.priority_tree,
                display,
                ("tree", "parent_id", "weight", "share", "expected_bytes", "actual_bytes", "deviation"),
            )

        def _fill_waits(self, rows: list[dict[str, Any]]) -> None:
            self._clear_and_fill(
                self.wait_tree,
                rows,
                ("kind", "wait_id", "reason", "streams", "start_ms", "end_ms", "closed_by"),
            )

        def _fill_snapshots(self, rows: list[dict[str, Any]]) -> None:
            display = []
            self._snapshot_paths = {}
            for row in rows:
                error = row.get("error")
                display.append(
                    {
                        **row,
                        "error": error.get("code") if isinstance(error, dict) else "",
                    }
                )
            children = self.snapshot_tree.get_children("")
            if children:
                self.snapshot_tree.delete(*children)
            for index, row in enumerate(display):
                iid = f"snapshot-{index}"
                self.snapshot_tree.insert(
                    "",
                    "end",
                    iid=iid,
                    values=tuple(
                        "" if row.get(key) is None else row.get(key)
                        for key in (
                            "name",
                            "label",
                            "declaration_version",
                            "strategy",
                            "script_index",
                            "content_digest",
                            "error",
                        )
                    ),
                )
                if row.get("path"):
                    self._snapshot_paths[iid] = str(row["path"])

        def _draw_windows(self) -> None:
            canvas = self.window_canvas
            canvas.delete("all")
            width = max(400, canvas.winfo_width())
            height = max(240, canvas.winfo_height())
            left, right, top, bottom = 56, 22, 24, 38
            plot_w = width - left - right
            plot_h = height - top - bottom
            series = self.view.get("window_series", {})
            points = series.get("connection", [])
            all_values: list[int] = []
            for point in points:
                for key in ("send_window", "recv_window"):
                    if isinstance(point.get(key), int):
                        all_values.append(point[key])
            for stream in series.get("streams", []):
                for point in stream.get("points", []):
                    for key in ("send_window", "recv_window"):
                        if isinstance(point.get(key), int):
                            all_values.append(point[key])
            max_y = max(all_values or [1])
            max_x = max((int(point.get("frame_seq") or 0) for point in points), default=1)
            canvas.create_line(left, top, left, top + plot_h, fill="#777777")
            canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="#777777")
            canvas.create_text(left, 8, text="window bytes", anchor="w", fill="#444444")
            canvas.create_text(left + plot_w, top + plot_h + 18, text="frame #", anchor="e", fill="#444444")

            def coords(point: dict[str, Any], key: str) -> tuple[float, float] | None:
                value = point.get(key)
                frame_seq = point.get("frame_seq")
                if not isinstance(value, int) or not isinstance(frame_seq, int):
                    return None
                x = left + (frame_seq / max(1, max_x)) * plot_w
                y = top + plot_h - (value / max(1, max_y)) * plot_h
                return x, y

            palette = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")
            legend_y = 10
            self._line_series(canvas, points, "send_window", "#111111", 2, coords)
            self._line_series(canvas, points, "recv_window", "#999999", 1, coords, dash=(4, 3))
            canvas.create_text(width - right, legend_y, text="conn send", anchor="e", fill="#111111")
            canvas.create_text(width - right, legend_y + 16, text="conn recv", anchor="e", fill="#999999")
            offset = 40
            for index, stream in enumerate(series.get("streams", [])):
                color = palette[index % len(palette)]
                self._line_series(canvas, stream.get("points", []), "send_window", color, 1, coords)
                legend_y = 10 + offset
                canvas.create_text(
                    width - right,
                    legend_y,
                    text=f"stream {stream['stream_id']} send",
                    anchor="e",
                    fill=color,
                )
                self._line_series(
                    canvas,
                    stream.get("points", []),
                    "recv_window",
                    color,
                    1,
                    coords,
                    dash=(2, 3),
                )
                canvas.create_text(
                    width - right,
                    legend_y + 14,
                    text=f"stream {stream['stream_id']} recv",
                    anchor="e",
                    fill=color,
                )
                offset += 30
            if not points:
                canvas.create_text(left + plot_w / 2, top + plot_h / 2, text="No frames yet", fill="#777777")

        def _line_series(
            self,
            canvas: Any,
            points: list[dict[str, Any]],
            key: str,
            color: str,
            width: int,
            coord_fn: Any,
            dash: tuple[int, int] | None = None,
        ) -> None:
            coords: list[float] = []
            for point in points:
                pair = coord_fn(point, key)
                if pair is not None:
                    coords.extend(pair)
            if len(coords) >= 4:
                canvas.create_line(
                    *coords,
                    fill=color,
                    width=width,
                    smooth=False,
                    dash=dash,
                )
            elif len(coords) == 2:
                canvas.create_oval(coords[0] - 2, coords[1] - 2, coords[0] + 2, coords[1] + 2, fill=color)


def main() -> int:
    if tk is None:
        print("Tkinter is not available in this Python installation.")
        print("On Windows, install the standard Python distribution and ensure the Tcl/Tk feature is selected.")
        return 2
    app = H2MuxApp()
    app.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
