"""Tkinter diagnostic workbench.

The module deliberately imports no Tk symbols at import time so CLI and test
runs still work on Python installations built without Tk support.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_IO, E_SNAPSHOT
from .injection import prepare_injected_inputs, write_digest_mismatch_snapshot
from .manifest import load_manifests
from .persistence import Persistence
from .reporting import invalidation_lines, pod_lines
from .restore import replay_payload, restore_payload
from .scheduler import WorkbenchSession


def _default_paths() -> tuple[str, str, str]:
    root = Path(__file__).resolve().parent.parent / "samples"
    return str(root / "nodes.json"), str(root / "pods.json"), str(root / "plugins.json")


def _set_text(widget: Any, text: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", text)
    widget.configure(state="disabled")


def launch_gui(args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise BusinessError(
            E_IO,
            "Tkinter could not be imported; install the official Windows Python 3.11+ distribution",
            details={"error": str(exc)},
        ) from None

    class DiagnosticApp:
        def __init__(self, root: Any) -> None:
            self.root = root
            self.tk = tk
            self.ttk = ttk
            self.messagebox = messagebox
            self.filedialog = filedialog
            self.workdir = Path(args.workdir)
            self.persistence = Persistence(self.workdir)
            default_nodes, default_pods, default_plugins = _default_paths()
            self.nodes_path = tk.StringVar(value=args.nodes or default_nodes)
            self.pods_path = tk.StringVar(value=args.pods or default_pods)
            self.plugins_path = tk.StringVar(value=args.plugins or default_plugins)
            self.snapshot_name = tk.StringVar(value="gui-snapshot")
            self.node_choice = tk.StringVar()
            self.label_key = tk.StringVar(value="disk")
            self.label_value = tk.StringVar(value="hdd")
            self.versions_text = tk.StringVar(value="no session")
            self.status_text = tk.StringVar(value="Ready")
            self.session: WorkbenchSession | None = None
            self._inject_once = getattr(args, "inject", "none")
            self._build()
            self._initial_load()

        def _build(self) -> None:
            self.root.title("Kubernetes Scheduling & Affinity Diagnostics")
            self.root.geometry("1440x900")
            self.root.minsize(1100, 700)
            style = self.ttk.Style(self.root)
            if "vista" in style.theme_names():
                style.theme_use("vista")
            style.configure("Status.TLabel", padding=4)

            outer = self.ttk.Frame(self.root, padding=8)
            outer.grid(row=0, column=0, sticky="nsew")
            self.root.rowconfigure(0, weight=1)
            self.root.columnconfigure(0, weight=1)
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(2, weight=1)

            files = self.ttk.LabelFrame(outer, text="Manifests and session", padding=6)
            files.grid(row=0, column=0, sticky="ew", pady=(0, 6))
            for column in (1, 4, 7):
                files.columnconfigure(column, weight=1)
            self.ttk.Label(files, text="Nodes").grid(row=0, column=0, sticky="w")
            self.ttk.Entry(files, textvariable=self.nodes_path).grid(row=0, column=1, sticky="ew", padx=4)
            self.ttk.Button(files, text="Browse", command=lambda: self._browse(self.nodes_path)).grid(row=0, column=2)
            self.ttk.Label(files, text="Pods").grid(row=0, column=3, sticky="w", padx=(8, 0))
            self.ttk.Entry(files, textvariable=self.pods_path).grid(row=0, column=4, sticky="ew", padx=4)
            self.ttk.Button(files, text="Browse", command=lambda: self._browse(self.pods_path)).grid(row=0, column=5)
            self.ttk.Label(files, text="Plugins").grid(row=0, column=6, sticky="w", padx=(8, 0))
            self.ttk.Entry(files, textvariable=self.plugins_path).grid(row=0, column=7, sticky="ew", padx=4)
            self.ttk.Button(files, text="Browse", command=lambda: self._browse(self.plugins_path)).grid(row=0, column=8)
            self.ttk.Button(files, text="Load + Full Schedule", command=self.load_and_schedule).grid(
                row=0, column=9, padx=(8, 0)
            )
            self.ttk.Button(files, text="Save Snapshot", command=self.save_snapshot).grid(row=0, column=10, padx=(8, 0))
            self.ttk.Entry(files, textvariable=self.snapshot_name, width=18).grid(row=0, column=11, padx=(8, 0))
            self.ttk.Button(files, text="Verify Full", command=self.verify_incremental).grid(row=0, column=12, padx=(8, 0))
            self.ttk.Label(files, textvariable=self.versions_text, style="Status.TLabel").grid(
                row=1, column=0, columnspan=13, sticky="w", pady=(5, 0)
            )

            controls = self.ttk.LabelFrame(outer, text="Incremental node-label edit", padding=6)
            controls.grid(row=1, column=0, sticky="ew", pady=(0, 6))
            self.ttk.Label(controls, text="Node").grid(row=0, column=0)
            self.node_menu = self.ttk.OptionMenu(controls, self.node_choice, "")
            self.node_menu.grid(row=0, column=1, sticky="w", padx=4)
            self.ttk.Label(controls, text="Label key").grid(row=0, column=2, padx=(8, 0))
            self.ttk.Entry(controls, textvariable=self.label_key, width=18).grid(row=0, column=3, padx=4)
            self.ttk.Label(controls, text="Value").grid(row=0, column=4, padx=(8, 0))
            self.ttk.Entry(controls, textvariable=self.label_value, width=18).grid(row=0, column=5, padx=4)
            self.ttk.Button(controls, text="Apply + Recompute", command=self.apply_label).grid(row=0, column=6, padx=4)
            self.ttk.Button(controls, text="Remove Key", command=self.remove_label).grid(row=0, column=7, padx=4)
            self.ttk.Label(controls, text="Snapshots are immutable and read-only", foreground="#555").grid(
                row=0, column=8, padx=(16, 0)
            )

            paned = self.ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
            paned.grid(row=2, column=0, sticky="nsew")

            left = self.ttk.Frame(paned)
            left.rowconfigure(0, weight=1)
            left.columnconfigure(0, weight=1)
            self.pod_tree = self.ttk.Treeview(
                left,
                columns=("queue", "pod", "selected", "version", "score", "error", "candidates"),
                show="headings",
                selectmode="browse",
            )
            headings = {
                "queue": "Queue",
                "pod": "Pod",
                "selected": "Selected",
                "version": "Version",
                "score": "Score milli",
                "error": "Error bound",
                "candidates": "Candidates",
            }
            widths = {"queue": 55, "pod": 130, "selected": 110, "version": 70, "score": 90, "error": 80, "candidates": 220}
            for column in self.pod_tree["columns"]:
                self.pod_tree.heading(column, text=headings[column])
                self.pod_tree.column(column, width=widths[column], anchor="w", stretch=column == "candidates")
            scrollbar = self.ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.pod_tree.yview)
            self.pod_tree.configure(yscrollcommand=scrollbar.set)
            self.pod_tree.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            self.pod_tree.bind("<<TreeviewSelect>>", self._show_selected_pod)
            paned.add(left, weight=2)

            right = self.ttk.Frame(paned)
            right.rowconfigure(0, weight=3)
            right.rowconfigure(1, weight=2)
            right.columnconfigure(0, weight=1)
            detail_frame = self.ttk.LabelFrame(right, text="Filter / Score detail (stable order)", padding=4)
            detail_frame.grid(row=0, column=0, sticky="nsew")
            detail_frame.rowconfigure(0, weight=1)
            detail_frame.columnconfigure(0, weight=1)
            self.detail = tk.Text(detail_frame, wrap="none", font=("Consolas", 9), state="disabled")
            detail_scroll_y = self.ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail.yview)
            detail_scroll_x = self.ttk.Scrollbar(detail_frame, orient=tk.HORIZONTAL, command=self.detail.xview)
            self.detail.configure(yscrollcommand=detail_scroll_y.set, xscrollcommand=detail_scroll_x.set)
            self.detail.grid(row=0, column=0, sticky="nsew")
            detail_scroll_y.grid(row=0, column=1, sticky="ns")
            detail_scroll_x.grid(row=1, column=0, sticky="ew")

            invalidation_frame = self.ttk.LabelFrame(right, text="Latest invalidation closure", padding=4)
            invalidation_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
            invalidation_frame.rowconfigure(0, weight=1)
            invalidation_frame.columnconfigure(0, weight=1)
            self.invalidation_text = tk.Text(invalidation_frame, wrap="none", font=("Consolas", 9), state="disabled", height=10)
            invalidation_scroll = self.ttk.Scrollbar(
                invalidation_frame, orient=tk.VERTICAL, command=self.invalidation_text.yview
            )
            self.invalidation_text.configure(yscrollcommand=invalidation_scroll.set)
            self.invalidation_text.grid(row=0, column=0, sticky="nsew")
            invalidation_scroll.grid(row=0, column=1, sticky="ns")
            paned.add(right, weight=3)

            snapshots = self.ttk.LabelFrame(outer, text="Snapshots", padding=4)
            snapshots.grid(row=3, column=0, sticky="ew", pady=(6, 0))
            snapshots.columnconfigure(0, weight=1)
            self.snapshot_tree = self.ttk.Treeview(
                snapshots,
                columns=("name", "revision", "digest", "versions", "path"),
                show="headings",
                height=5,
                selectmode="browse",
            )
            snapshot_headings = {
                "name": "Name",
                "revision": "Rev",
                "digest": "Content digest",
                "versions": "Versions",
                "path": "Read-only backup",
            }
            for column in self.snapshot_tree["columns"]:
                self.snapshot_tree.heading(column, text=snapshot_headings[column])
                self.snapshot_tree.column(column, width=150 if column in {"digest", "path"} else 90, anchor="w")
            self.snapshot_tree.grid(row=0, column=0, sticky="ew")
            self.ttk.Button(snapshots, text="Restore Selected + Replay", command=self.restore_selected).grid(
                row=0, column=1, padx=(6, 0), sticky="n"
            )

            status = self.ttk.Label(outer, textvariable=self.status_text, style="Status.TLabel", relief="sunken")
            status.grid(row=4, column=0, sticky="ew", pady=(6, 0))
            self.refresh_snapshots()

        def _browse(self, variable: Any) -> None:
            path = self.filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
            if path:
                variable.set(path)

        def _error(self, error: BusinessError, event: str) -> None:
            self.persistence.log_error(error, event=event, injection=self._inject_once)
            location = f" at {error.location}" if error.location else ""
            self.status_text.set(f"{error.code}: {error.message}{location}")
            self.messagebox.showerror("Diagnostic error", f"{error.code}: {error.message}{location}")

        def _initial_load(self) -> None:
            try:
                if self._inject_once == "snapshot-digest":
                    self.load_and_schedule(injection="none")
                    snapshots = self.persistence.list_snapshots()
                    if not snapshots:
                        raise BusinessError(E_SNAPSHOT, "snapshot digest injection requires an existing snapshot")
                    latest = self.persistence.load_snapshot(snapshots[-1]["name"], snapshots[-1]["revision"])
                    altered = write_digest_mismatch_snapshot(latest, self.workdir)
                    self.persistence.log("WARN", "fault_injection", kind="snapshot-digest", file=str(altered))
                    self.persistence.load_snapshot_file(altered)
                else:
                    self.load_and_schedule(injection=self._inject_once)
            except BusinessError as exc:
                self._error(exc, "gui_initial_load_failed")
            finally:
                self._inject_once = "none"

        def load_and_schedule(self, injection: str = "none") -> None:
            try:
                nodes_path, pods_path, plugins_path, injection_meta = prepare_injected_inputs(
                    self.nodes_path.get(),
                    self.pods_path.get(),
                    self.plugins_path.get() or None,
                    self.workdir,
                    injection,
                )
                if injection_meta["kind"] != "none":
                    self.persistence.log("WARN", "fault_injection", **injection_meta)
                nodes, pods, plugins = load_manifests(nodes_path, pods_path, plugins_path)
                session = WorkbenchSession(nodes, pods, plugins)
                session.full_schedule()
                self.session = session
                self.persistence.save_current(session)
                self.persistence.log(
                    "INFO",
                    "gui_full_schedule",
                    digest=session.content_digest(),
                    versions=session.versions(),
                )
                self._rebuild_node_menu()
                self._render_all()
                self.status_text.set("Full schedule completed; incremental edit is ready.")
            except BusinessError as exc:
                self._error(exc, "gui_full_schedule_failed")

        def _rebuild_node_menu(self) -> None:
            if self.session is None:
                return
            menu = self.node_menu["menu"]
            menu.delete(0, "end")
            names = sorted(self.session.nodes)
            for name in names:
                menu.add_command(label=name, command=lambda value=name: self.node_choice.set(value))
            if names:
                self.node_choice.set(names[0])

        def _render_versions(self) -> None:
            if self.session is None:
                self.versions_text.set("no session")
                return
            versions = self.session.versions()
            self.versions_text.set(
                f"versions node={versions['node']} pod={versions['pod']} plugin={versions['plugin']} "
                f"graph={versions['graph']} | digest={self.session.content_digest()} | "
                f"graph={len(self.session.graph.nodes)} nodes/{sum(len(v) for v in self.session.graph.edges.values())} edges"
            )

        def _render_pods(self) -> None:
            for item in self.pod_tree.get_children():
                self.pod_tree.delete(item)
            if self.session is None:
                return
            for index, pod in enumerate(self.session.pods):
                result = self.session.results[pod.pod_id]
                item = self.pod_tree.insert(
                    "",
                    "end",
                    iid=pod.pod_id,
                    values=(
                        index,
                        pod.pod_id,
                        result["selected"] or "<none>",
                        result["version"],
                        result.get("selected_score_milli"),
                        result["error_bound_milli"],
                        ",".join(result["candidates"]),
                    ),
                )
                if index == 0:
                    self.pod_tree.selection_set(item)

        def _render_invalidation(self) -> None:
            if self.session is None or self.session.last_invalidation is None:
                _set_text(self.invalidation_text, "No invalidation yet.")
                return
            _set_text(self.invalidation_text, "\n".join(invalidation_lines(self.session.last_invalidation, max_steps=100)))

        def _render_all(self) -> None:
            self._render_versions()
            self._render_pods()
            self._render_invalidation()
            self._show_selected_pod()
            self.refresh_snapshots()

        def _show_selected_pod(self, _event: Any = None) -> None:
            if self.session is None:
                _set_text(self.detail, "No session loaded.")
                return
            selection = self.pod_tree.selection()
            if not selection:
                _set_text(self.detail, "Select a pod to inspect candidates and scores.")
                return
            pod_id = selection[0]
            if pod_id not in self.session.results:
                return
            result = self.session.results[pod_id]
            lines = [
                *pod_lines(self.session, pod_id, include_all_scores=True),
                "",
                f"plugin_weights={result['plugin_weights']}",
                f"score_scale={result['score_scale']} truncation={result['truncation']}",
            ]
            _set_text(self.detail, "\n".join(lines))

        def apply_label(self) -> None:
            self._mutate_label(delete=False)

        def remove_label(self) -> None:
            self._mutate_label(delete=True)

        def _mutate_label(self, *, delete: bool) -> None:
            if self.session is None:
                self.status_text.set("Load a session before editing labels.")
                return
            node = self.node_choice.get()
            key = self.label_key.get().strip()
            value = None if delete else self.label_value.get()
            try:
                report = self.session.edit_node_label(node, key, value)
                self.persistence.save_current(self.session)
                self.persistence.log(
                    "INFO",
                    "gui_incremental_label",
                    node=node,
                    key=key,
                    value=value,
                    affectedPods=report.get("pods", []),
                    affectedStats=len(report.get("stats", [])),
                )
                self._render_all()
                self.status_text.set(
                    f"Incremental recompute: {len(report.get('pods', []))} pod(s), "
                    f"{len(report.get('stats', []))} stat item(s); "
                    f"unaffected={self.session._unaffected_check['count'] if self.session._unaffected_check else 0}"
                )
            except BusinessError as exc:
                self._error(exc, "gui_incremental_label_failed")

        def verify_incremental(self) -> None:
            if self.session is None:
                self.status_text.set("Load a session first.")
                return
            try:
                report = self.session.verify_full_consistency()
                self.persistence.log("INFO", "gui_incremental_verify", **report)
                self.status_text.set(
                    f"Incremental equals full recomputation: pods={report['podCount']} stats={report['statCount']}"
                )
            except BusinessError as exc:
                self._error(exc, "gui_incremental_verify_failed")

        def save_snapshot(self) -> None:
            if self.session is None:
                self.status_text.set("Load a session first.")
                return
            try:
                summary = self.persistence.save_snapshot(self.session, self.snapshot_name.get())
                self.refresh_snapshots()
                self.status_text.set(
                    f"Snapshot {summary['name']}@r{summary['revision']} saved read-only at {summary['backupPath']}"
                )
            except BusinessError as exc:
                self._error(exc, "gui_snapshot_save_failed")

        def refresh_snapshots(self) -> None:
            for item in self.snapshot_tree.get_children():
                self.snapshot_tree.delete(item)
            try:
                snapshots = self.persistence.list_snapshots()
            except BusinessError:
                return
            for item in snapshots:
                self.snapshot_tree.insert(
                    "",
                    "end",
                    iid=f"{item['name']}@r{item['revision']}",
                    values=(
                        item["name"],
                        item["revision"],
                        item["digest"],
                        item["versions"],
                        item["backupPath"],
                    ),
                )

        def restore_selected(self) -> None:
            selection = self.snapshot_tree.selection()
            if not selection:
                self.status_text.set("Select a snapshot first.")
                return
            item_id = selection[0]
            try:
                name, revision_text = item_id.rsplit("@r", 1)
                payload = self.persistence.load_snapshot(name, int(revision_text))
                if self._inject_once == "snapshot-digest":
                    altered = write_digest_mismatch_snapshot(payload, self.workdir)
                    self.persistence.log("WARN", "fault_injection", kind="snapshot-digest", file=str(altered))
                    payload = self.persistence.load_snapshot_file(altered)
                restored, report = restore_payload(payload, current_session=self.session, source=item_id)
                replay = replay_payload(payload, source=item_id)
                self.session = restored
                self.persistence.save_current(restored)
                self.persistence.log(
                    "INFO",
                    "gui_snapshot_restore",
                    snapshot=item_id,
                    replaceRequired=report["replaceRequired"],
                    changedNodes=report["changedNodes"],
                    changedPods=report["changedPods"],
                    replay=replay,
                )
                self._rebuild_node_menu()
                self._render_all()
                self.status_text.set(
                    f"Restored {item_id}; replaceRequired={report['replaceRequired']} "
                    f"changedNodes={report['changedNodes']} changedPods={report['changedPods']}; "
                    f"replay={replay['ok']} affected={replay['replayedAffectedPods']}"
                )
            except BusinessError as exc:
                self._error(exc, "gui_snapshot_restore_failed")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise BusinessError(
            E_IO,
            "Tkinter could not open a display",
            details={"error": str(exc)},
        ) from None
    DiagnosticApp(root)
    root.mainloop()
    return 0
