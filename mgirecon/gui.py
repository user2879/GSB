"""Tkinter diagnostics interface, imported lazily for headless environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .commands import apply_commands
from .config import config_hash, load_config
from .engine import reconstruct
from .errors import E_GUI_UNAVAILABLE, diagnostic
from .report import build_report
from .snapshot import load_snapshot, write_snapshot


def _gui_unavailable(exc: Exception) -> dict[str, Any]:
    return diagnostic(
        E_GUI_UNAVAILABLE,
        f"Tkinter is not available in this runtime: {exc}",
        stage="gui",
        detail={"hint": "WSL Ubuntu: sudo apt install python3-tk"},
    )


def gui_capability() -> dict[str, Any]:
    try:
        import tkinter

        return {
            "ok": True,
            "error": None,
            "tk_version": tkinter.TkVersion,
            "display": __import__("os").environ.get("DISPLAY"),
        }
    except Exception as exc:
        return {"ok": False, "error": _gui_unavailable(exc)}


def launch_gui(project_dir: str | Path, result: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        return {"ok": False, "error": _gui_unavailable(exc)}

    root_dir = Path(project_dir)
    config = load_config(root_dir)

    class DiagnosticsWindow(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("MG-Recon 增量诊断")
            self.geometry("1180x780")
            self.current_result = result
            self._build()
            if self.current_result is None:
                self.refresh_incremental()
            else:
                self.render(self.current_result)

        def _build(self) -> None:
            toolbar = ttk.Frame(self, padding=6)
            toolbar.pack(fill="x")
            ttk.Label(toolbar, text="残差阈值").pack(side="left")
            self.threshold_var = tk.StringVar(value=str(config["reconstruction"]["residual_threshold"]))
            ttk.Entry(toolbar, textvariable=self.threshold_var, width=7).pack(side="left", padx=(4, 12))
            ttk.Label(toolbar, text="最小视差").pack(side="left")
            self.disparity_var = tk.StringVar(value=str(config["reconstruction"]["min_disparity"]))
            ttk.Entry(toolbar, textvariable=self.disparity_var, width=5).pack(side="left", padx=(4, 12))
            ttk.Label(toolbar, text="量化档位").pack(side="left")
            self.quant_var = tk.StringVar(value=str(config["reconstruction"]["quantization_level"]))
            ttk.Entry(toolbar, textvariable=self.quant_var, width=5).pack(side="left", padx=(4, 12))
            ttk.Label(toolbar, text="体素尺寸").pack(side="left")
            self.voxel_size_var = tk.StringVar(value=str(config["reconstruction"]["voxel_size"]))
            ttk.Entry(toolbar, textvariable=self.voxel_size_var, width=5).pack(side="left", padx=(4, 12))
            ttk.Button(toolbar, text="编辑并增量重算", command=self.edit_and_run).pack(side="left")
            ttk.Button(toolbar, text="刷新", command=self.refresh_incremental).pack(side="left", padx=6)

            notebook = ttk.Notebook(self)
            notebook.pack(fill="both", expand=True, padx=6, pady=6)
            overview = ttk.Frame(notebook)
            matches = ttk.Frame(notebook)
            voxels = ttk.Frame(notebook)
            ledger = ttk.Frame(notebook)
            diagnostics = ttk.Frame(notebook)
            notebook.add(overview, text="视图与预算")
            notebook.add(matches, text="匹配对")
            notebook.add(voxels, text="自适应体素")
            notebook.add(ledger, text="失效与复用")
            notebook.add(diagnostics, text="待办与异常")

            self.view_tree = ttk.Treeview(
                overview,
                columns=("view", "features", "depth", "boundary", "pairs"),
                show="headings",
            )
            for column, width in (("view", 80), ("features", 100), ("depth", 110), ("boundary", 110), ("pairs", 360)):
                self.view_tree.heading(column, text=column)
                self.view_tree.column(column, width=width)
            self.view_tree.pack(fill="both", expand=True)

            self.match_tree = ttk.Treeview(
                matches,
                columns=("pair", "inliers", "selected", "rank", "tied", "tie_winner"),
                show="headings",
            )
            for column in ("pair", "inliers", "selected", "rank", "tied", "tie_winner"):
                self.match_tree.heading(column, text=column)
                self.match_tree.column(column, width=130)
            self.match_tree.pack(fill="both", expand=True)

            self.voxel_text = tk.Text(voxels, wrap="word", height=28)
            self.voxel_text.pack(fill="both", expand=True)
            self.ledger_text = tk.Text(ledger, wrap="none", height=28)
            self.ledger_text.pack(fill="both", expand=True)
            self.diag_text = tk.Text(diagnostics, wrap="word", height=28)
            self.diag_text.pack(fill="both", expand=True)

        def render(self, result: dict[str, Any]) -> None:
            report = build_report(result, project_dir=root_dir)
            for item in self.view_tree.get_children():
                self.view_tree.delete(item)
            for view in report["views"]:
                self.view_tree.insert(
                    "",
                    "end",
                    values=(
                        view["view"],
                        view["features"],
                        view["depth_pixels"],
                        view["boundary_pixels"],
                        ", ".join(view["matching_pairs"]) or "-",
                    ),
                )
            for item in self.match_tree.get_children():
                self.match_tree.delete(item)
            for pair in report["match_pairs"]:
                self.match_tree.insert(
                    "",
                    "end",
                    values=(
                        pair["pair"],
                        pair["inliers"],
                        pair["selected"],
                        pair["rank"],
                        pair["tied"],
                        pair["tie_winner"],
                    ),
                )
            voxel = report["voxel"]
            self.voxel_text.delete("1.0", "end")
            self.voxel_text.insert(
                "1.0",
                "子体素统计\n"
                + f"总数: {voxel.get('voxel_count', 0)}\n"
                + f"叶节点: {voxel.get('leaf_count', 0)}\n"
                + f"细分节点: {voxel.get('subdivided_count', 0)}\n"
                + f"最大细分深度: {voxel.get('max_depth_used', 0)}\n"
                + f"细分深度分布: {voxel.get('depth_distribution', {})}\n"
                + f"邻接边: {voxel.get('adjacency_edges', 0)}\n"
                + f"策略: {report['strategy'].get('name')} / depth={report['strategy'].get('effective_max_depth')} / order={report['strategy'].get('split_order')}\n",
            )
            self.ledger_text.delete("1.0", "end")
            ledger = report["ledger"]
            self.ledger_text.insert(
                "1.0",
                f"复用合计: {ledger.get('reused_total', 0)} {ledger.get('reused_by_stage', {})}\n"
                + f"失效合计: {ledger.get('invalidated_total', 0)} {ledger.get('invalidated_by_stage', {})}\n"
                + f"失效关系: {ledger.get('invalidated_by_relation', {})}\n\n",
            )
            for entry in result.get("ledger", {}).get("invalidated", [])[:120]:
                self.ledger_text.insert(
                    "end",
                    f"[失效] {entry['stage']} {entry['key']} {entry['relations']} {entry['reason']}\n",
                )
            for entry in result.get("ledger", {}).get("reused", [])[:120]:
                self.ledger_text.insert("end", f"[复用] {entry['stage']} {entry['key']} {entry['cache_key'][:12]}\n")
            self.diag_text.delete("1.0", "end")
            self.diag_text.insert("1.0", f"预算: {report['budget']}\n\n待办队列:\n")
            for item in report["pending"]:
                self.diag_text.insert("end", f"  {item}\n")
            self.diag_text.insert("end", "\n失败清单:\n")
            for error in report["errors"]:
                self.diag_text.insert("end", f"  [{error['code']}] {error['stage']} {error['message']}\n")

        def edit_and_run(self) -> None:
            try:
                threshold = int(self.threshold_var.get())
                disparity = int(self.disparity_var.get())
                quant_level = int(self.quant_var.get())
                voxel_size = int(self.voxel_size_var.get())
            except ValueError:
                messagebox.showerror("参数错误", "阈值、最小视差、量化档位和体素尺寸必须是整数")
                return
            commands = [
                {"op": "set_param", "name": "residual_threshold", "value": threshold},
                {"op": "set_param", "name": "min_disparity", "value": disparity},
                {"op": "set_param", "name": "quantization_level", "value": quant_level},
                {"op": "set_param", "name": "voxel_size", "value": voxel_size},
            ]
            updated, errors = apply_commands(root_dir, load_config(root_dir), commands)
            if errors:
                messagebox.showerror("编辑失败", "\n".join(f"{item['code']}: {item['message']}" for item in errors))
                return
            snapshot = load_snapshot(root_dir, config_hash(updated))
            rebuilt = reconstruct(root_dir, updated, previous_state=snapshot["state"], inherited_errors=errors)
            manifest = write_snapshot(root_dir, rebuilt, previous_manifest=snapshot["manifest"])
            rebuilt["manifest"] = manifest
            self.current_result = rebuilt
            self.render(rebuilt)

        def refresh_incremental(self) -> None:
            current_config = load_config(root_dir)
            snapshot = load_snapshot(root_dir, config_hash(current_config))
            rebuilt = reconstruct(root_dir, current_config, previous_state=snapshot["state"])
            manifest = write_snapshot(root_dir, rebuilt, previous_manifest=snapshot["manifest"])
            rebuilt["manifest"] = manifest
            self.current_result = rebuilt
            self.render(rebuilt)

    try:
        window = DiagnosticsWindow()
        window.mainloop()
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": _gui_unavailable(exc)}
