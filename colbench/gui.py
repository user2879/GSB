"""Tkinter 可视化诊断工作台。

Windows 官方 Python 安装包默认包含 tkinter，可直接 ``python -m colbench.gui``
打开窗口；在无显示环境（或本沙箱缺少 tkinter）中自动降级为
:class:`colbench.view.DashboardModel.render_text` 文本渲染，渲染顺序保持一致。
"""
from __future__ import annotations

import os
from pathlib import Path

from .view import DashboardModel


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    display = os.environ.get("DISPLAY")
    if os.name != "nt" and not display:
        return False
    return True


def launch(model: DashboardModel, *, force_text: bool = False) -> str:
    """启动工作台；返回实际使用的渲染模式 ``tk`` 或 ``text``。"""
    if force_text or not tkinter_available():
        output = model.render_text()
        print(output)
        return "text"
    return _launch_tk(model)


def _launch_tk(model: DashboardModel) -> str:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("列式存储压缩与扫描诊断工作台")
    root.geometry("980x760")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    def add_tab(title: str, lines: list[str]) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        text = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 10))
        scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                 command=text.yview)
        scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL,
                                 command=text.xview)
        text.configure(yscrollcommand=scroll_y.set,
                       xscrollcommand=scroll_x.set)
        text.grid(row=0, column=0, sticky=tk.NSEW)
        scroll_y.grid(row=0, column=1, sticky=tk.NS)
        scroll_x.grid(row=1, column=0, sticky=tk.EW)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)

    summary = model.plan.summary()
    add_tab("参数版本与业务码", [
        f"budget={model.version.budget} encoding={model.version.encoding} "
        f"format={model.version.format}",
        "业务码: " + (", ".join(model.business_codes)
                      if model.business_codes else "OK"),
    ])
    add_tab("列裁剪与谓词下推", [
        f"计划摘要: {summary['plan_digest']}",
        "投影列: " + ", ".join(summary["projected_columns"]),
        "裁掉列: " + ", ".join(summary["pruned_columns"]),
        "下推: " + ", ".join(str(item) for item in summary["pushed"]),
        "保留: " + ", ".join(str(item) for item in summary["retained"]),
        f"候选块 {summary['candidate_block_count']} / 跳过块 "
        f"{summary['skipped_block_count']}",
        *[f"依据: {reason}" for reason in summary["reasons"]],
    ])
    add_tab("块编码与压缩比", [
        f"{row['segment_id']} rg={row['row_group']} {row['column']} "
        f"{row['encoding']} compressed={row['compressed']} "
        f"pre={row['pre_bytes']} post={row['post_bytes']} "
        f"ratio={row['pre_bytes'] / max(1, row['post_bytes']):.2f}"
        for row in model.block_rows])
    add_tab("任务调度时间线", [
        f"{task.task_id} {task.segment_id} rg={task.row_group} "
        f"{task.column} {task.status} wait={task.wait_ms}ms "
        f"run={task.run_ms}ms mem={task.memory_cost}B"
        for task in model.schedule.tasks])
    add_tab("每段配额占用", [
        f"{row['segment_id']} peak={row['peak_running']} "
        f"quota={row['quota']} tasks={row['tasks']}"
        for row in model.segment_quota])
    add_tab("扫描统计", [
        f"{key}: {value}" for key, value in
        model.schedule.statistics().items()])
    replacement_lines = []
    if model.replacement:
        report = model.replacement.to_dict()
        replacement_lines = [
            f"清空轨迹 {report['cleared_trajectory_count']} 条",
            f"受影响块 {len(report['affected_block_keys'])} 个",
            *[f"受影响块: {key}" for key in report["affected_block_keys"]],
        ]
    add_tab("替换与快照", replacement_lines
            + [f"快照: {name}" for name in model.snapshots])
    add_tab("权威重算", [str(model.verify_report or "未执行")])

    root.mainloop()
    return "tk"


def write_text_report(model: DashboardModel, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(model.render_text(), encoding="utf-8")
    return path
