from __future__ import annotations

import json
from typing import Any

from .errors import DiagError, E_ENV
from .runtime import DiagnosticRuntime


class DiagnosticConsole:
    """Tkinter console built without importing Tk at module import time."""

    def __init__(self, root: Any, runtime: DiagnosticRuntime, tk: Any, ttk: Any) -> None:
        self.root = root
        self.runtime = runtime
        self.tk = tk
        self.ttk = ttk
        self.tree_item_ids: dict[str, str] = {}
        self.status_var = tk.StringVar(value=f"root={runtime.view.root_revision} ready")
        self.root_var = tk.StringVar(value=self._root_label())
        self._build()
        self.render_full()
        self.refresh_stats()

    def _build(self) -> None:
        tk = self.tk
        ttk = self.ttk
        self.root.title("Cluster Config Distribution Diagnostics")
        self.root.geometry("1180x720")
        self.root.minsize(900, 520)

        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, textvariable=self.root_var, font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        toolbar = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Query selected cluster", command=self.query_selected).pack(side="left")
        ttk.Button(toolbar, text="Commit next revision", command=self.commit_next).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Authoritative recompute", command=self.authoritative).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Recover + replay", command=self.recover).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Close", command=self.root.destroy).pack(side="right")

        self.progress = ttk.Progressbar(toolbar, mode="determinate", maximum=100, length=170)
        self.progress.pack(side="right", padx=(0, 10))

        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        tree_frame = ttk.Frame(paned)
        detail_frame = ttk.Frame(paned)
        paned.add(tree_frame, weight=3)
        paned.add(detail_frame, weight=2)

        columns = ("cluster", "revision", "task", "parent", "digest", "status")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        widths = {"cluster": 150, "revision": 72, "task": 140, "parent": 72, "digest": 190, "status": 130}
        for column in columns:
            self.tree.heading(column, text=column.capitalize())
            self.tree.column(column, width=widths[column], anchor="w", stretch=column in {"task", "digest"})
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.show_selected_summary)

        ttk.Label(detail_frame, text="Result / diagnostics", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.detail = tk.Text(detail_frame, wrap="word", height=22)
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail.yview)
        self.detail.configure(yscrollcommand=detail_scroll.set, state="disabled")
        self.detail.pack(side="left", fill="both", expand=True)
        detail_scroll.pack(side="right", fill="y")

        self.stats = tk.Text(self.root, wrap="none", height=8)
        self.stats.configure(state="disabled")
        self.stats.pack(fill="x", padx=10, pady=(0, 10))
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _write_detail(self, title: str, payload: Any) -> None:
        self.detail.configure(state="normal")
        self.detail.insert("end", f"\n[{title}]\n")
        if isinstance(payload, str):
            self.detail.insert("end", payload + "\n")
        else:
            self.detail.insert(
                "end",
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            )
        self.detail.see("end")
        self.detail.configure(state="disabled")

    def _root_label(self) -> str:
        root_revision = self.runtime.view.root_revision
        snapshot = self.runtime.view.active_snapshot
        digest = snapshot.snapshot_digest or "none"
        return f"ROOT revision {root_revision} | snapshot {digest[:24]}..."

    def _row_values(self, item: Any) -> tuple[str, int, str, str, str, str]:
        return (
            item.cluster_id,
            item.node_revision,
            item.task_id,
            "" if item.parent_revision is None else str(item.parent_revision),
            item.content_digest[:26] + "...",
            f"current @root {item.root_revision}",
        )

    def _ensure_root_item(self) -> None:
        values = ("ROOT", self.runtime.view.root_revision, "-", "-", self._root_label().split("snapshot ", 1)[-1], "single-root binding")
        if self.tree.exists("root"):
            self.tree.item("root", values=values, open=True)
        else:
            self.tree.insert("", "end", iid="root", values=values, open=True)

    def render_full(self) -> None:
        for item_id in self.tree.get_children(""):
            self.tree.delete(item_id)
        self.tree_item_ids.clear()
        self._ensure_root_item()
        for item in self.runtime.view.items:
            tree_id = f"cluster:{item.cluster_id}"
            self.tree.insert("root", "end", iid=tree_id, values=self._row_values(item))
            self.tree_item_ids[item.cluster_id] = tree_id
        self.root_var.set(self._root_label())

    def apply_root_switch(self, report: Any) -> None:
        self._ensure_root_item()
        current_clusters = {item.cluster_id: item for item in self.runtime.view.items}
        for cluster_id in list(self.tree_item_ids):
            if cluster_id not in current_clusters:
                self.tree.delete(self.tree_item_ids.pop(cluster_id))
        touched = set(report.incremental_rows_touched)
        for cluster_id, item in current_clusters.items():
            tree_id = self.tree_item_ids.get(cluster_id, f"cluster:{cluster_id}")
            if cluster_id in touched or not self.tree.exists(tree_id):
                if self.tree.exists(tree_id):
                    self.tree.item(tree_id, values=self._row_values(item))
                else:
                    self.tree.insert("root", "end", iid=tree_id, values=self._row_values(item))
                self.tree_item_ids[cluster_id] = tree_id
            else:
                # Unchanged node payload remains untouched; only its single-root binding label moves.
                self.tree.set(tree_id, "status", f"current @root {item.root_revision}")
        self.root_var.set(self._root_label())
        changed_lines = []
        for change in report.changed_clusters:
            changed_lines.append(
                f"{change['cluster_id']}: r{change['old_revision']}->{change['new_revision']} "
                f"{str(change['old_summary_digest'])[:18]} -> {str(change['new_summary_digest'])[:18]}"
            )
        stale_lines = [f"{node['cluster_id']}@r{node['revision']} (bound r{node['bound_root_revision']})" for node in report.stale_nodes]
        self._write_detail(
            "ROOT SWITCH",
            {
                "old_root": report.old_root_revision,
                "new_root": report.new_root_revision,
                "affected_clusters": list(report.affected_clusters),
                "changed_clusters": changed_lines,
                "stale_nodes": stale_lines,
                "policy": "whole replacement; stale items excluded from current statistics",
            },
        )
        self.refresh_stats()

    def show_selected_summary(self, _event: Any = None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        tree_id = selection[0]
        if not tree_id.startswith("cluster:"):
            return
        cluster_id = tree_id.split(":", 1)[1]
        item = self.runtime.view.item(cluster_id)
        if item is None:
            return
        self._write_detail(
            f"NODE {cluster_id} revision {item.node_revision}",
            item.aggregate_summary,
        )

    def refresh_stats(self) -> None:
        stats = self.runtime.latency_stats()
        self.stats.configure(state="normal")
        self.stats.delete("1.0", "end")
        self.stats.insert(
            "end",
            f"clock={stats['clock_basis']} truth={stats['truth_source']} unit={stats['unit']} "
            f"quant_error<={stats['quantization_error_upper_bound_ms_x1000']} (ms_x1000)\n",
        )
        if not stats["groups"]:
            self.stats.insert("end", "No latency observations bound to the current root.\n")
        for group in stats["groups"]:
            self.stats.insert(
                "end",
                f"{group['kind']:<13} cluster={str(group['cluster_id']):<12} rev={str(group['revision']):<4} "
                f"n={group['count']} mean={group['mean_ms_x1000']} p95={group['p95_ms_x1000']} "
                f"max={group['max_ms_x1000']} exceeded={group['exceeded_count']}\n",
            )
        self.stats.configure(state="disabled")

    def _selected_cluster(self) -> str | None:
        selection = self.tree.selection()
        if not selection or not selection[0].startswith("cluster:"):
            return None
        return selection[0].split(":", 1)[1]

    def query_selected(self) -> None:
        cluster_id = self._selected_cluster()
        if cluster_id is None:
            self._write_detail("QUERY", "Select a current cluster row first.")
            return
        self._set_status(f"querying {cluster_id}...")
        try:
            result = self.runtime.query(cluster_id)
        except DiagError as exc:
            self._set_status(f"query failed: {exc.code}")
            self._write_detail("QUERY ERROR", exc.as_dict())
            return
        self.progress["value"] = 100
        latency = result["latency"]
        status = f"query {cluster_id} r{result['revision']} {latency['elapsed_ms_x1000']} ms_x1000"
        if latency["exceeded"]:
            status += f" | E_LATENCY over={latency['overage_ms_x1000']} budget={latency['budget_ms_x1000']}"
        self._set_status(status)
        self._write_detail("QUERY RESULT", result)
        self.refresh_stats()

    def _progress_callback(self, state: str, cluster_id: str, revision: int, details: dict[str, Any]) -> None:
        if state == "revision_started":
            self.progress["value"] = 10
            self._set_status(f"revision {revision}: starting {len(details.get('clusters', []))} cluster node(s)")
        elif state == "cluster_started":
            self.progress["value"] = 55
            self._set_status(f"revision {revision}: {cluster_id} running {details.get('task_id')}")
        elif state == "cluster_completed":
            self.progress["value"] = 80
            self._set_status(f"revision {revision}: {cluster_id} completed")
        elif state == "revision_completed":
            self.progress["value"] = 100
            self._set_status(f"revision {revision}: committed and root replaced")

    def commit_next(self) -> None:
        self.progress["value"] = 0
        self._set_status("committing next revision...")
        try:
            result = self.runtime.commit_next(progress=self._progress_callback)
        except DiagError as exc:
            self._set_status(f"commit failed: {exc.code}")
            self._write_detail("COMMIT ERROR", exc.as_dict())
            return
        report = self.runtime.view.last_switch
        if report is not None:
            self.apply_root_switch(report)
        self._write_detail(
            "COMMIT COMPLETE",
            {
                "revision": result["committed_revision"],
                "nodes": result["nodes"],
                "unknown_uncommitted_nodes": result["unknown_uncommitted_nodes"],
            },
        )

    def authoritative(self) -> None:
        self._set_status("authoritative whole-tree recomputation running...")
        try:
            result = self.runtime.authoritative()
        except DiagError as exc:
            self._set_status(f"authoritative verification failed: {exc.code}")
            self._write_detail("AUTHORITATIVE ERROR", exc.as_dict())
            return
        self._set_status(f"authoritative OK: {result['recomputed_node_count']} node(s) matched")
        self._write_detail("AUTHORITATIVE OK", result)
        self.refresh_stats()

    def recover(self) -> None:
        self._set_status("recovering snapshot and replaying nodes...")
        try:
            result = self.runtime.recover(replay=True)
        except DiagError as exc:
            self._set_status(f"recovery failed: {exc.code}")
            self._write_detail("RECOVERY ERROR", exc.as_dict())
            return
        if result["snapshot_replaced"]:
            self.render_full()
        self._set_status(f"recovery OK: replay={result['replay']['replayed_node_count'] if result['replay'] else 'skipped'}")
        self._write_detail("RECOVERY + REPLAY", result)
        self.refresh_stats()


def run_gui(runtime: DiagnosticRuntime) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise DiagError(E_ENV, f"Tkinter is unavailable: {exc}", field_path="tkinter") from exc
    root = tk.Tk()
    DiagnosticConsole(root, runtime, tk, ttk)
    root.mainloop()
