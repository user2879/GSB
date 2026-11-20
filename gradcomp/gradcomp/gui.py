"""Tkinter 可视化界面。

展示各工作进程通信阶段、已收发块数、该步量化误差界、误差归因排序、
全局梯度一致性状态与快照列表，所有列表按固定键/固定顺序渲染。
无显示环境或 Tkinter 缺失时，build_headless_state 仍可产出界面数据，
供自动化测试与 CI 无头验证。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .engine import train
from .snapshot import list_snapshots


def build_headless_state(cfg: Config, workdir: str | Path) -> dict[str, Any]:
    """不依赖 Tkinter，生成与界面渲染同源的稳定状态数据。"""
    report = train(cfg, workdir=workdir)
    listing = list_snapshots(workdir)
    worker_rows: list[dict[str, Any]] = []
    # 固定按 rank 升序，保证渲染顺序稳定。
    for step in report.steps:
        for worker in sorted(step["workers"], key=lambda w: w["rank"]):
            worker_rows.append({
                "step": step["step"],
                "rank": worker["rank"],
                "phases": " -> ".join(worker["counters"]["phases"]),
                "sent_blocks": worker["counters"]["sent_blocks"],
                "recv_blocks": worker["counters"]["recv_blocks"],
                "global_digest": worker["global_gradient_digest"],
            })

    error_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    for step in report.steps:
        bounds = step["error_bounds"]
        if "single_value_bound" in bounds:
            error_rows.append({
                "step": step["step"],
                "single_value_bound": bounds["single_value_bound"],
                "vector_relative_bound":
                    bounds["vector_relative_bound"]["vector_relative_bound"],
                "measured_truncation_error_2norm":
                    bounds["measured_truncation_error_2norm"],
                "vector_bound_respected": bounds["vector_bound_respected"],
                "scale": bounds["depends_on"]["scale"],
                "observed_max_abs":
                    bounds["depends_on"]["observed_gradient_range"]["max_abs"],
            })
        for item in step["attribution"]["top_elements"]:
            attribution_rows.append({"step": step["step"], **item})
        consistency_rows.append({
            "step": step["step"],
            "consistent": step["consistent"],
            "reference_digest": step["reference_digest"],
            "worker_digests": step["worker_global_digests"],
            "first_diff_index": step["first_diff_index"],
        })

    return {
        "config": cfg.summary(),
        "workers": worker_rows,
        "error_bounds": error_rows,
        "attribution": attribution_rows,
        "consistency": consistency_rows,
        "snapshots": listing["snapshots"],
        "pending_files": listing["pending_files"],
        "final_parameters_digest": report.final_parameters_digest,
        "final_parameter_version": report.final_parameter_version,
    }


def launch_gui(cfg: Config, workdir: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # 无头 Linux 常见
        raise SystemExit(
            "Tkinter 不可用（Windows 官方 Python 安装包默认提供，"
            f"安装时勾选 tcl/tk）。可改用 headless 状态导出。详情: {exc}")

    state = build_headless_state(cfg, workdir)
    root = tk.Tk()
    root.title("梯度压缩通信仿真台")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    def _make_table(parent: tk.Misc, columns: list[str],
                    rows: list[dict[str, Any]]) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col in columns:  # 列顺序即渲染顺序，稳定不变。
            tree.heading(col, text=col)
            tree.column(col, width=160, anchor="w")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        for row in rows:
            tree.insert("", "end", values=[
                _format(row.get(col)) for col in columns])

    tabs = [
        ("工作进程", ["step", "rank", "phases", "sent_blocks",
                      "recv_blocks", "global_digest"], state["workers"]),
        ("误差界", ["step", "single_value_bound", "vector_relative_bound",
                    "measured_truncation_error_2norm",
                    "vector_bound_respected", "scale", "observed_max_abs"],
         state["error_bounds"]),
        ("误差归因", ["step", "element", "dominant_source",
                      "truncation_error", "order_error", "abs_contribution",
                      "share_of_total"], state["attribution"]),
        ("一致性", ["step", "consistent", "reference_digest",
                    "first_diff_index"], state["consistency"]),
        ("快照", ["file", "completed_steps", "parameter_version",
                  "snapshot_version", "last_step_complete",
                  "parameters_digest"], state["snapshots"]),
    ]
    for title, columns, rows in tabs:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        _make_table(tab, columns, rows)

    status = ttk.Label(
        root,
        text=f"最终参数版本 {state['final_parameter_version']}  "
             f"摘要 {state['final_parameters_digest'][:16]}  "
             f"待决文件 {len(state['pending_files'])}")
    status.pack(side="bottom", fill="x")
    root.mainloop()


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(v)[:12] for v in value)
    return str(value)
