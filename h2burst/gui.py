"""Tkinter desktop workbench.

Tkinter is imported lazily so headless validation can import the package and
exercise the pure view model on machines without a Tk runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .errors import WorkbenchError
from .viewmodel import WorkbenchViewModel


def launch_gui(
    session_path: str | Path | None = None,
    *,
    snapshot_db: str | Path = "snapshots.sqlite3",
) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    class WorkbenchApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("HTTP/2 Burst Window Diagnostic Workbench")
            self.root.geometry("1440x900")
            self.root.minsize(1000, 640)
            self.view_model: WorkbenchViewModel | None = None
            self.snapshot_db = str(snapshot_db)
            self._build_ui()
            if session_path is not None:
                try:
                    self.open_session(str(session_path))
                except WorkbenchError as exc:
                    self.status.configure(text=str(exc))
                    messagebox.showerror(exc.code, str(exc), parent=self.root)
            self.root.protocol("WM_DELETE_WINDOW", self.close)

        def _build_ui(self) -> None:
            style = ttk.Style(self.root)
            if "vista" in style.theme_names():
                style.theme_use("vista")
            elif "clam" in style.theme_names():
                style.theme_use("clam")

            toolbar = ttk.Frame(self.root, padding=(8, 6))
            toolbar.pack(fill="x")
            buttons: list[tuple[str, Callable[[], None]]] = [
                ("Open session", self.choose_session),
                ("Run", self.run_all),
                ("Step event", self.step_event),
                ("Snapshot", self.save_snapshot),
                ("Restore", self.restore_snapshot),
                ("Replay", self.replay_snapshot),
                ("Refresh", self.refresh),
            ]
            for label, handler in buttons:
                ttk.Button(toolbar, text=label, command=self._guard(handler)).pack(
                    side="left", padx=(0, 6)
                )
            self.session_label = ttk.Label(toolbar, text="No session loaded")
            self.session_label.pack(side="right")

            self.strategy_label = ttk.Label(
                self.root,
                text="strategy: not evaluated",
                anchor="w",
                padding=(10, 4),
            )
            self.strategy_label.pack(fill="x")

            paned = ttk.Panedwindow(self.root, orient="vertical")
            paned.pack(fill="both", expand=True, padx=8, pady=(0, 6))
            self.notebook = ttk.Notebook(paned)
            paned.add(self.notebook, weight=4)
            self.trees: dict[str, ttk.Treeview] = {}

            frame_columns = (
                ("frame", 58),
                ("event", 58),
                ("type", 82),
                ("stream", 64),
                ("bytes", 72),
                ("conn_before", 92),
                ("conn_after", 92),
                ("stream_before", 98),
                ("stream_after", 98),
                ("strategy", 100),
                ("business", 116),
            )
            self._add_tree_tab("Frames", "frames", frame_columns)
            stream_columns = (
                ("stream", 62),
                ("priority", 62),
                ("state", 126),
                ("body_remaining", 104),
                ("stream_window", 104),
                ("conn_window", 96),
                ("actual_bytes", 96),
                ("expected_bytes", 104),
                ("gap_rounds", 82),
            )
            self._add_tree_tab("Streams / windows", "streams", stream_columns)
            transition_columns = (
                ("index", 56),
                ("stream", 62),
                ("event", 62),
                ("frame", 62),
                ("from", 128),
                ("to", 128),
                ("reason", 280),
            )
            self._add_tree_tab("Transitions", "transitions", transition_columns)
            fairness_columns = (
                ("stream", 62),
                ("priority", 62),
                ("expected", 100),
                ("actual", 100),
                ("deviation", 100),
                ("ratio", 100),
                ("state", 126),
            )
            self._add_tree_tab("Fairness", "fairness", fairness_columns)
            warning_columns = (
                ("type", 168),
                ("stream", 62),
                ("priority", 62),
                ("gap_rounds", 78),
                ("start_event", 82),
                ("source_event", 86),
                ("beneficiary", 82),
                ("deviation_ratio", 102),
            )
            self._add_tree_tab("Starvation", "warnings", warning_columns)
            snapshot_columns = (
                ("id", 55),
                ("label", 250),
                ("version", 70),
                ("next_event", 82),
                ("digest", 120),
                ("read_only", 76),
            )
            self._add_tree_tab("Snapshots", "snapshots", snapshot_columns, selectmode="browse")

            detail_frame = ttk.LabelFrame(paned, text="Selected frame / stream detail", padding=6)
            paned.add(detail_frame, weight=1)
            self.detail = tk.Text(detail_frame, height=7, wrap="none", state="disabled")
            detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail.yview)
            self.detail.configure(yscrollcommand=detail_scroll.set)
            self.detail.pack(side="left", fill="both", expand=True)
            detail_scroll.pack(side="right", fill="y")
            self.trees["frames"].bind("<<TreeviewSelect>>", self._show_frame_detail)

            self.status = ttk.Label(self.root, text="ready", anchor="w", padding=(10, 4))
            self.status.pack(fill="x")
            self.root.bind("<F5>", lambda _event: self.refresh())
            self.root.bind("<Control-o>", lambda _event: self.choose_session())

        def _add_tree_tab(
            self,
            title: str,
            name: str,
            columns: tuple[tuple[str, int], ...],
            *,
            selectmode: str = "extended",
        ) -> None:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            tree = ttk.Treeview(
                frame,
                columns=[column for column, _width in columns],
                show="headings",
                selectmode=selectmode,
            )
            for column, width in columns:
                tree.heading(column, text=column.replace("_", " "))
                tree.column(column, width=width, minwidth=45, stretch=column in {"reason", "label"})
            y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            self.trees[name] = tree

        def _guard(self, action: Callable[[], None]) -> Callable[[], None]:
            def wrapped() -> None:
                try:
                    action()
                except WorkbenchError as exc:
                    self.status.configure(text=str(exc))
                    messagebox.showerror(exc.code, str(exc), parent=self.root)
                except Exception as exc:  # defensive UI boundary
                    self.status.configure(text=f"unexpected error: {exc}")
                    messagebox.showerror("Unexpected error", str(exc), parent=self.root)

            return wrapped

        def ensure_view_model(self) -> WorkbenchViewModel | None:
            if self.view_model is None:
                messagebox.showinfo(
                    "Open a session",
                    "Load a local session JSON file first.",
                    parent=self.root,
                )
                return None
            return self.view_model

        def open_session(self, path: str) -> None:
            if self.view_model is not None:
                self.view_model.close()
            self.view_model = WorkbenchViewModel.from_file(
                path,
                snapshot_db=self.snapshot_db,
                auto_run=False,
            )
            self.session_label.configure(text=path)
            self.refresh()

        def choose_session(self) -> None:
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Open HTTP/2 session declaration",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            )
            if path:
                self.open_session(path)

        def run_all(self) -> None:
            view_model = self.ensure_view_model()
            if view_model is not None:
                view_model.run()
                self.refresh()

        def step_event(self) -> None:
            view_model = self.ensure_view_model()
            if view_model is not None:
                view_model.step()
                self.refresh()

        def save_snapshot(self) -> None:
            view_model = self.ensure_view_model()
            if view_model is None:
                return
            default = f"event-{view_model.engine.event_cursor}"
            label = simpledialog.askstring(
                "Save snapshot",
                "Snapshot label:",
                initialvalue=default,
                parent=self.root,
            )
            if label:
                view_model.save_snapshot(label)
                self.refresh()

        def selected_snapshot_id(self) -> int | None:
            selected = self.trees["snapshots"].selection()
            if not selected:
                messagebox.showinfo("Select a snapshot", "Select a snapshot row first.", parent=self.root)
                return None
            return int(selected[0])

        def restore_snapshot(self) -> None:
            view_model = self.ensure_view_model()
            if view_model is None:
                return
            snapshot_id = self.selected_snapshot_id()
            if snapshot_id is None:
                return
            result = view_model.restore_snapshot(snapshot_id)
            self.refresh()
            if result.whole_replacement:
                messagebox.showinfo(
                    "Whole-session replacement",
                    f"Snapshot {snapshot_id} replaced the current session.\n"
                    f"added={len(result.diff['added'])} removed={len(result.diff['removed'])} "
                    f"changed={len(result.diff['changed'])}\n"
                    f"backup_snapshot={result.backup_snapshot_id}",
                    parent=self.root,
                )

        def replay_snapshot(self) -> None:
            view_model = self.ensure_view_model()
            if view_model is None:
                return
            snapshot_id = self.selected_snapshot_id()
            if snapshot_id is None:
                return
            result = view_model.replay_snapshot(snapshot_id)
            self.refresh()
            messagebox.showinfo(
                "Replay verified",
                f"Compared {result.frames_compared} frames and "
                f"{result.transitions_compared} transitions.\n"
                f"digest={result.replay_digest}",
                parent=self.root,
            )

        def _populate(self, name: str, rows: list[dict[str, Any]]) -> None:
            tree = self.trees[name]
            selection = set(tree.selection())
            tree.delete(*tree.get_children())
            for index, row in enumerate(rows):
                iid = str(row.get("frame", row.get("index", index)))
                if name == "snapshots":
                    iid = str(row["id"])
                values = [row.get(column, "") for column in tree["columns"]]
                tree.insert("", "end", iid=iid, values=values)
            for iid in selection:
                if tree.exists(iid):
                    tree.selection_add(iid)

        def refresh(self) -> None:
            view_model = self.view_model
            if view_model is None:
                self.status.configure(text="Load a session JSON file to begin.")
                return
            self._populate("frames", view_model.frame_rows())
            self._populate("streams", view_model.stream_rows())
            self._populate("transitions", view_model.transition_rows())
            self._populate("fairness", view_model.fairness_rows())
            self._populate("warnings", view_model.warning_rows())
            self._populate("snapshots", view_model.snapshot_rows())
            self.strategy_label.configure(text=f"strategy: {view_model.strategy_text()}")
            status = view_model.status_rows()
            self.status.configure(
                text=(
                    f"session={status['session']} conn_window={status['connection_window']} "
                    f"frames={status['frames']} transitions={status['transitions']} "
                    f"warnings={status['warnings']} next_event={status['next_event_seq']} | "
                    f"{status['status_code']}: {status['status_message']}"
                )
            )

        def _show_frame_detail(self, _event: object | None = None) -> None:
            if self.view_model is None:
                return
            selected = self.trees["frames"].selection()
            if not selected:
                return
            try:
                frame_seq = int(selected[0])
            except ValueError:
                return
            frame = next((item for item in self.view_model.engine.frames if item.frame_seq == frame_seq), None)
            if frame is None:
                return
            text = "\n".join(
                f"{key}: {value}" for key, value in frame.to_dict().items()
            )
            self.detail.configure(state="normal")
            self.detail.delete("1.0", "end")
            self.detail.insert("1.0", text)
            self.detail.configure(state="disabled")

        def close(self) -> None:
            if self.view_model is not None:
                self.view_model.close()
                self.view_model = None
            self.root.destroy()

    root = tk.Tk()
    WorkbenchApp(root)
    root.mainloop()
