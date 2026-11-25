from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .editor import apply_command
from .engine import build_reconstruction, canonical_state_blob
from .snapshot import SnapshotStore
from .validation import load_project


class DiagnosticApp:
    def __init__(self, root: tk.Tk, project_root: Path):
        self.root = root
        self.project_root = project_root
        self.store = SnapshotStore(project_root)
        self.last_state = self.store.load_latest_state()
        root.title("MVRV 多分辨率体素占位诊断")
        root.geometry("1080x720")

        self.metric_vars: dict[str, tk.StringVar] = {}
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="项目根目录").grid(row=0, column=0, sticky=tk.W)
        self.root_var = tk.StringVar(value=str(project_root))
        ttk.Entry(controls, textvariable=self.root_var, width=48).grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="重新校验/加载", command=self.reload_project).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="全量重建", command=self.full_rebuild).grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="增量重算", command=self.incremental_rebuild).grid(row=0, column=4, padx=4)

        self.param_vars = {
            "min_parallax": tk.StringVar(value="2"),
            "residual_threshold_q": tk.StringVar(value="820"),
            "voxel_size": tk.StringVar(value="4"),
        }
        for column, (key, variable) in enumerate(self.param_vars.items()):
            ttk.Label(controls, text=key).grid(row=1, column=column * 2, sticky=tk.W, pady=(8, 0))
            ttk.Entry(controls, textvariable=variable, width=12).grid(row=1, column=column * 2 + 1, sticky=tk.W, pady=(8, 0))
        ttk.Button(controls, text="应用参数并增量重算", command=self.apply_parameters).grid(row=1, column=6, padx=8, pady=(8, 0))
        ttk.Button(controls, text="替换视图1", command=self.replace_view).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=6)
        ttk.Button(controls, text="修改视图2外参", command=self.change_extrinsic).grid(row=2, column=2, columnspan=2, sticky=tk.W)
        ttk.Button(controls, text="快照恢复判定", command=self.recover_snapshot).grid(row=2, column=4, columnspan=2, sticky=tk.W)

        metrics_frame = ttk.LabelFrame(root, text="关键指标", padding=8)
        metrics_frame.pack(fill=tk.X, padx=8)
        metric_names = [
            "strategy_id", "view_count", "pair_count", "feature_count", "match_count",
            "triangulated_points", "depth_boundary_pixels", "depth_conflict_pixels",
            "voxel_count", "occupied_voxels", "subdivided_voxels", "max_voxel_level",
        ]
        for index, name in enumerate(metric_names):
            row, column = divmod(index, 4)
            ttk.Label(metrics_frame, text=name).grid(row=row * 2, column=column, sticky=tk.W, padx=8)
            variable = tk.StringVar(value="-")
            self.metric_vars[name] = variable
            ttk.Label(metrics_frame, textvariable=variable, font=("Monospace", 11, "bold")).grid(
                row=row * 2 + 1, column=column, sticky=tk.W, padx=8, pady=(0, 6)
            )

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.tree = ttk.Treeview(body, columns=("value",), show="tree headings")
        self.tree.heading("#0", text="账本/产物")
        self.tree.heading("value", text="值")
        body.add(self.tree, weight=1)

        log_frame = ttk.Frame(body)
        self.log = tk.Text(log_frame, wrap=tk.NONE, font=("Monospace", 10))
        scroll_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll_y.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        body.add(log_frame, weight=2)

        self.loaded = None
        self.reload_project()
        ledger = self.store.load_latest_ledger()
        if ledger:
            self._update_tree(ledger)

    def _set_root(self) -> None:
        self.project_root = Path(self.root_var.get()).expanduser().resolve()
        self.store = SnapshotStore(self.project_root)

    def reload_project(self) -> None:
        self._set_root()
        self.loaded = load_project(self.project_root)
        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, json.dumps(self.loaded.report(), ensure_ascii=False, indent=2) + "\n")
        if self.loaded.ok and self.last_state:
            self._update_metrics(self.last_state.get("metrics", {}))
            ledger = self.store.load_latest_ledger()
            if ledger:
                self._update_tree(ledger)

    def _run(self, previous: dict | None) -> None:
        if self.loaded is None or not self.loaded.ok:
            self.reload_project()
            if not self.loaded or not self.loaded.ok:
                return
        result = build_reconstruction(self.loaded, previous_state=previous, workers=2)
        self.last_state = result["state"]
        self.store.commit(validation_report=self.loaded.report(), state=self.last_state, ledger=result["ledger"])
        self._update_metrics(self.last_state["metrics"])
        self._update_tree(result["ledger"])
        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, f"state_hash={canonical_state_blob(self.last_state)}\n")
        self.log.insert(tk.END, json.dumps(result["ledger"]["counts"], ensure_ascii=False, indent=2, sort_keys=True))

    def full_rebuild(self) -> None:
        self._run(None)

    def incremental_rebuild(self) -> None:
        previous = self.last_state or self.store.load_latest_state()
        self._run(previous)

    def apply_parameters(self) -> None:
        config_path = self.project_root / "config" / "reconstruction.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key, variable in self.param_vars.items():
            config[key] = int(variable.get())
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        self.reload_project()
        self.incremental_rebuild()

    def replace_view(self) -> None:
        command = {
            "op": "replace_view",
            "view_index": 1,
            "source": "replacements/view_01.png",
        }
        apply_command(self.project_root, command)
        self.reload_project()
        self.incremental_rebuild()

    def change_extrinsic(self) -> None:
        apply_command(self.project_root, {"op": "set_extrinsic", "view_index": 2, "t": [-3, -1, 0]})
        self.reload_project()
        self.incremental_rebuild()

    def recover_snapshot(self) -> None:
        report = self.store.recover()
        self.log.delete("1.0", tk.END)
        self.log.insert(tk.END, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    def _update_metrics(self, metrics: dict) -> None:
        for key, variable in self.metric_vars.items():
            variable.set(str(metrics.get(key, "-")))

    def _update_tree(self, ledger: dict) -> None:
        self.tree.delete(*self.tree.get_children())
        for section, value in ledger.get("counts", {}).items():
            self.tree.insert("", tk.END, text=section, values=(value,))
        for section in ("features", "pairs", "points", "voxels"):
            node = self.tree.insert("", tk.END, text=section, values=("",))
            for key, value in ledger.get(section, {}).items():
                preview = value if isinstance(value, (int, str)) else f"{len(value)} items"
                self.tree.insert(node, tk.END, text=key, values=(preview,))


def launch(project_root: Path | None = None) -> None:
    root = tk.Tk()
    DiagnosticApp(root, Path(project_root or Path.cwd()))
    root.mainloop()
