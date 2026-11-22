"""Tkinter diagnostic workbench.

The UI is a strict read-only view over ledger-backed conclusions: refreshing or
switching selection never writes to SQLite. Only explicit import, recompute and
snapshot actions mutate the ledger. All list rendering uses fixed sort keys so
repeated refreshes produce identical output.
"""
from __future__ import annotations

import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

from .constants import PLUGIN_LABELS, PLUGIN_ORDER
from .errors import SimError
from .scheduler.engine import ScheduleReport, SessionParams, default_params
from .session import Session

WINDOW_TITLE = "K8S Scheduling & Affinity Diagnostic Workbench"


class DiagnosticApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1180x760")
        self.session: Optional[Session] = None
        self.report: Optional[ScheduleReport] = None
        self._build_layout()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Import nodes...",
                   command=self._import_nodes).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Import pods...",
                   command=self._import_pods).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Recompute",
                   command=self._recompute).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Save snapshot",
                   command=self._save_snapshot).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Refresh view",
                   command=self._refresh_view).pack(side=tk.LEFT, padx=4)

        param_frame = ttk.LabelFrame(self.root, text="Session parameters", padding=8)
        param_frame.pack(fill=tk.X, padx=8)
        self.scale_var = tk.StringVar(value=str(default_params().scale_factor))
        ttk.Label(param_frame, text="scale factor").pack(side=tk.LEFT)
        ttk.Entry(param_frame, textvariable=self.scale_var, width=10).pack(
            side=tk.LEFT, padx=(4, 16)
        )
        self.weight_vars: Dict[str, tk.StringVar] = {}
        for plugin in PLUGIN_ORDER:
            ttk.Label(param_frame, text=PLUGIN_LABELS[plugin]).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default_params().weights[plugin]))
            self.weight_vars[plugin] = var
            ttk.Entry(param_frame, textvariable=var, width=7).pack(
                side=tk.LEFT, padx=(4, 12)
            )

        self.status_var = tk.StringVar(value="No session loaded.")
        ttk.Label(self.root, textvariable=self.status_var, foreground="#335").pack(
            anchor=tk.W, padx=10, pady=(6, 0)
        )

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(body)
        body.add(left, weight=1)
        columns = ("pod", "feasible", "bound", "tie", "error", "version")
        self.pod_tree = ttk.Treeview(left, columns=columns, show="headings",
                                     selectmode="browse")
        for key, heading, width in (
            ("pod", "Pod", 120),
            ("feasible", "Feasible", 80),
            ("bound", "Bound node", 120),
            ("tie", "Tie reason", 130),
            ("error", "Error influenced", 120),
            ("version", "Param version", 110),
        ):
            self.pod_tree.heading(key, text=heading)
            self.pod_tree.column(key, width=width, anchor=tk.W)
        self.pod_tree.pack(fill=tk.BOTH, expand=True)
        self.pod_tree.bind("<<TreeviewSelect>>", self._show_pod_detail)

        right = ttk.PanedWindow(body, orient=tk.VERTICAL)
        body.add(right, weight=3)

        score_frame = ttk.LabelFrame(right, text="Candidate scores and error bounds")
        right.add(score_frame, weight=3)
        score_cols = ["node", "total_mp", "points", "error_mp", "error_points"]
        score_cols.extend(PLUGIN_ORDER)
        self.score_tree = ttk.Treeview(
            score_frame, columns=score_cols, show="headings"
        )
        headings = {
            "node": ("Node", 100),
            "total_mp": ("Total (micro-pts)", 130),
            "points": ("Total points", 100),
            "error_mp": ("Error bound mp", 120),
            "error_points": ("Error pts", 90),
        }
        for plugin in PLUGIN_ORDER:
            headings[plugin] = (PLUGIN_LABELS[plugin], 140)
        for key in score_cols:
            text, width = headings[key]
            self.score_tree.heading(key, text=text)
            self.score_tree.column(key, width=width, anchor=tk.W)
        self.score_tree.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        right.add(bottom, weight=2)

        reject_frame = ttk.LabelFrame(bottom, text="Filter rejections (gate order)")
        bottom.add(reject_frame, weight=1)
        self.reject_text = tk.Text(reject_frame, height=8, wrap=tk.WORD)
        self.reject_text.pack(fill=tk.BOTH, expand=True)

        snapshot_frame = ttk.LabelFrame(bottom, text="Snapshots")
        bottom.add(snapshot_frame, weight=1)
        snap_cols = ("snapshot_id", "created_at", "version", "spec", "params")
        self.snapshot_tree = ttk.Treeview(
            snapshot_frame, columns=snap_cols, show="headings"
        )
        for key, heading, width in (
            ("snapshot_id", "Snapshot", 100),
            ("created_at", "Created", 140),
            ("version", "Version", 70),
            ("spec", "Spec digest", 110),
            ("params", "Params digest", 110),
        ):
            self.snapshot_tree.heading(key, text=heading)
            self.snapshot_tree.column(key, width=width, anchor=tk.W)
        self.snapshot_tree.pack(fill=tk.BOTH, expand=True)

        self._nodes_path: Optional[Path] = None
        self._pods_path: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _import_nodes(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self._nodes_path = Path(path)
            self._set_status(f"nodes file: {path}")

    def _import_pods(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self._pods_path = Path(path)
            self._set_status(f"pods file: {path}")

    def _ensure_session(self) -> Session:
        if not self._nodes_path or not self._pods_path:
            raise SimError("import both nodes and pods JSON files first")
        ledger_path = Path(tempfile.gettempdir()) / "k8ssim_workbench.sqlite"
        return Session.create(self._nodes_path, self._pods_path, ledger_path)

    def _read_params(self) -> SessionParams:
        base = self.session.params if self.session else default_params()
        weights = {plugin: int(var.get()) for plugin, var in self.weight_vars.items()}
        scale_factor = int(self.scale_var.get())
        return SessionParams(
            weights=weights,
            scale_factor=scale_factor,
            version=base.version,
        )

    def _recompute(self) -> None:
        try:
            if self.session is None:
                self.session = self._ensure_session()
            requested = self._read_params()
            self.session.update_parameters(
                weights=requested.weights,
                scale_factor=requested.scale_factor,
            )
            self.report = self.session.run_full_schedule()
            self._populate()
        except (SimError, ValueError) as exc:
            messagebox.showerror("Recompute failed", str(exc))

    def _save_snapshot(self) -> None:
        if not self.session or not self.report:
            messagebox.showinfo("Snapshot", "run a schedule before saving a snapshot")
            return
        snapshot_id = f"snap-v{self.session.params.version}"
        self.session.ledger.save_snapshot(
            snapshot_id, self.session.params, self.session.spec_digest()
        )
        self._populate_snapshots()
        self._set_status(f"snapshot {snapshot_id} saved")

    def _refresh_view(self) -> None:
        if self.report is not None:
            self._populate()

    # ------------------------------------------------------------------ #
    # Rendering (stable ordering, no ledger writes)
    # ------------------------------------------------------------------ #

    def _populate(self) -> None:
        assert self.session is not None and self.report is not None
        for item in self.pod_tree.get_children():
            self.pod_tree.delete(item)
        for decision in self.report.decisions:
            self.pod_tree.insert(
                "",
                tk.END,
                iid=decision.pod_id,
                values=(
                    decision.pod_id,
                    decision.feasible,
                    decision.bound_node or "-",
                    decision.tie_reason,
                    decision.error_influenced,
                    decision.parameter_version,
                ),
            )
        self._set_status(
            f"param version={self.session.params.version} "
            f"scale={self.session.params.scale_factor} "
            f"staleness={self.report.staleness}"
        )
        self._populate_snapshots()
        if self.report.decisions:
            first = self.report.decisions[0].pod_id
            self.pod_tree.selection_set(first)
            self._show_pod_detail()

    def _show_pod_detail(self, _event=None) -> None:
        selection = self.pod_tree.selection()
        if not selection or not self.report:
            return
        decision = next(
            item for item in self.report.decisions if item.pod_id == selection[0]
        )
        for item in self.score_tree.get_children():
            self.score_tree.delete(item)
        for node_score in decision.node_scores:
            values = [
                node_score.node,
                node_score.total_mp,
                node_score.total_points,
                node_score.error_bound_mp,
                node_score.error_bound_points,
            ]
            plugin_map = node_score.plugin_map()
            values.extend(
                f"{plugin_map[p].raw_score}/c{plugin_map[p].contribution_mp}"
                for p in PLUGIN_ORDER
            )
            self.score_tree.insert("", tk.END, values=values)

        self.reject_text.delete("1.0", tk.END)
        if not decision.rejections:
            self.reject_text.insert(tk.END, "(no rejections)")
        for rejection in decision.rejections:
            self.reject_text.insert(
                tk.END,
                f"{rejection.node} [{rejection.gate}] {rejection.reason} :: "
                f"{rejection.detail}\n",
            )
        if decision.error_influenced:
            self.reject_text.insert(
                tk.END,
                f"\nBinding is ERROR-INFLUENCED: gap={decision.score_gap_mp} mp "
                f"<= combined error={decision.combined_error_mp} mp; "
                f"runner-up={decision.runner_up}.",
            )

    def _populate_snapshots(self) -> None:
        if self.session is None:
            return
        for item in self.snapshot_tree.get_children():
            self.snapshot_tree.delete(item)
        for snap in self.session.snapshots():
            self.snapshot_tree.insert(
                "",
                tk.END,
                values=(
                    snap["snapshot_id"],
                    snap["created_at"],
                    snap["parameter_version"],
                    snap["spec_digest"][:12],
                    snap["params_digest"][:12],
                ),
            )

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)


def main() -> None:
    root = tk.Tk()
    DiagnosticApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
