"""Tkinter diagnostics console UI.

Only nodes affected by the current batch are updated in place (incremental
view); stale nodes outside the active root chain are listed read-only and
excluded from the statistics. The module imports Tkinter lazily so the rest
of the package works on headless machines.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from .engine import Engine, NodeKey
from .errors import DiagError
from .ledger import Ledger
from .model import load_spec
from .orchestrator import Orchestrator
from .snapshot import build_snapshot, save_snapshot


class HeadlessError(RuntimeError):
    pass


def _require_tk():
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        return tk, ttk, messagebox
    except Exception as exc:  # pragma: no cover - environment dependent
        raise HeadlessError(
            "Tkinter is unavailable in this environment; use the CLI on "
            f"headless machines. Underlying error: {exc}") from exc


class DiagnosticsUI:
    def __init__(self, samples: str | Path, data: str | Path,
                 use_services: bool = True):
        self.tk, self.ttk, self.messagebox = _require_tk()
        self.samples = Path(samples)
        self.data = Path(data)
        self.use_services = use_services
        self.spec = load_spec(self.samples / "clusters.json",
                              self.samples / "tasks.json",
                              self.samples / "revisions.json")
        self.ledger = Ledger(self.data / "ledger.db")
        self.engine = Engine(self.spec, self.ledger, None)
        self.events: queue.Queue = queue.Queue()
        self.node_rows: dict[tuple, str] = {}
        self.root_window = None

    def build(self):
        tk, ttk, _ = self.tk, self.ttk, self.messagebox
        self.root_window = tk.Tk()
        self.root_window.title("Multi-Cluster Configuration Diagnostics Console")
        self.root_window.geometry("1080x680")
        toolbar = ttk.Frame(self.root_window, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Apply Incremental",
                   command=self.on_apply).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Authority Rebuild",
                   command=self.on_rebuild).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Rollback r3",
                   command=self.on_rollback).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Refresh View",
                   command=self.refresh).pack(side="left", padx=4)
        self.status = tk.StringVar(value="ready")
        ttk.Label(toolbar, textvariable=self.status).pack(side="right", padx=8)

        columns = ("version", "task", "cluster", "total", "digest", "state")
        self.tree = ttk.Treeview(self.root_window, columns=columns,
                                 show="headings", selectmode="browse")
        headers = {"version": "Revision", "task": "Task",
                   "cluster": "Cluster", "total": "Aggregate Total",
                   "digest": "Content Digest", "state": "State"}
        widths = {"version": 70, "task": 140, "cluster": 140,
                  "total": 120, "digest": 420, "state": 90}
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=widths[col], anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)

        stats = ttk.LabelFrame(self.root_window, text="Statistics", padding=8)
        stats.pack(fill="x", padx=8, pady=4)
        self.stats_var = tk.StringVar(value="no active root")
        ttk.Label(stats, textvariable=self.stats_var).pack(anchor="w")
        self.root_window.after(200, self.drain_events)
        self.refresh()
        return self.root_window

    def run(self) -> None:
        window = self.build()
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.mainloop()

    def close(self) -> None:
        self.ledger.close()
        if self.root_window is not None:
            self.root_window.destroy()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        view = self.engine.active_view()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.node_rows.clear()
        for node in view["active"]:
            self._insert_row(node, "active")
        for node in view["isolated"]:
            self._insert_row(node, "isolated (read-only)")
        active = view["active"]
        totals = [n["aggregate"]["total"] for n in active]
        self.stats_var.set(
            f"root=r{view['root']}  active_nodes={len(active)}  "
            f"isolated_nodes={len(view['isolated'])}  "
            f"sum_of_totals={sum(totals)}")

    def _insert_row(self, node: dict, state: str) -> None:
        iid = self.tree.insert(
            "", "end",
            values=(node["revision"], node["task"], node["cluster"],
                    node["aggregate"]["total"], node["digest"][:18] + "...",
                    state))
        self.node_rows[(node["revision"], node["task"],
                        node["cluster"])] = iid

    def _upsert_row(self, key: NodeKey, state: str) -> None:
        stored = self.ledger.get_node(key.task_id, key.cluster_id, key.version)
        if stored is None:
            return
        import json
        aggregate = json.loads(stored["aggregate_json"])
        values = (key.version, key.task_id, key.cluster_id,
                  aggregate["total"], stored["content_digest"][:18] + "...",
                  state)
        iid = self.node_rows.get((key.version, key.task_id, key.cluster_id))
        if iid is None:
            iid = self.tree.insert("", "end", values=values)
            self.node_rows[(key.version, key.task_id, key.cluster_id)] = iid
        else:
            self.tree.item(iid, values=values)

    def drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, index, total, key = event
                    self.status.set(f"progress {index}/{total} "
                                    f"{key.task_id} {key.cluster_id}")
                    self._upsert_row(key, "updated")
                elif kind == "done":
                    self.status.set("completed")
                    self.refresh()
                elif kind == "error":
                    self.messagebox.showerror("Diagnostics Error",
                                              str(event[1]))
                    self.status.set("error")
        except queue.Empty:
            pass
        self.root_window.after(200, self.drain_events)

    def _background(self, job):
        def runner():
            try:
                job()
                self.events.put(("done",))
            except DiagError as exc:
                self.events.put(("error", exc))
        threading.Thread(target=runner, daemon=True).start()

    def on_apply(self) -> None:
        self.status.set("starting services and applying revisions...")
        self._background(self._apply_job)

    def _apply_job(self) -> None:
        orch = Orchestrator(self.spec) if self.use_services else None
        if orch is not None:
            orch.start_all()
        engine = Engine(self.spec, self.ledger, orch)
        try:
            state_path = self.data / "last_chain.json"
            import json
            previous = json.loads(state_path.read_text("utf-8")) \
                if state_path.exists() else None
            engine.apply_chain(previous, "ui-apply",
                               progress=lambda i, t, k:
                               self.events.put(("progress", i, t, k)))
            state_path.write_text(__import__(
                "diagconsole.errors", fromlist=["canonical_json"]).canonical_json(
                [{"version": r.version, "values": r.values,
                  "strategy": r.strategy, "derives": list(r.derives)}
                 for r in self.spec.revisions]), encoding="utf-8")
            head = self.spec.revisions[-1].version
            snapshot = build_snapshot(
                self.spec, head, self.ledger.list_nodes("committed"),
                {"stage": "ui", "head": head})
            save_snapshot(self.data / "snapshot.json", snapshot)
        finally:
            if orch is not None:
                orch.stop_all()

    def on_rebuild(self) -> None:
        self.status.set("authority rebuild from root...")
        self._background(lambda: self.engine.rebuild(
            self.spec.revisions[0].version, use_services=False,
            progress=lambda i, t, k:
            self.events.put(("progress", i, t, k))))

    def on_rollback(self) -> None:
        self.status.set("rolling back to revision 3...")
        self._background(lambda: self.engine.rollback(
            3, progress=lambda i, t, k:
            self.events.put(("progress", i, t, k))))


def launch(samples: str = "samples", data: str = "data",
           use_services: bool = True) -> None:
    DiagnosticsUI(samples, data, use_services).run()
