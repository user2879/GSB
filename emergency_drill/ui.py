"""Tkinter 诊断界面。

Tkinter 在模块导入时不强制加载，便于无显示环境（含部分 Linux CI）运行核心测试；
Windows 官方 Python 安装包自带 tkinter，直接启动窗口。刷新动作只读账本，不产生写入。
"""

from .viewmodel import DiagnosticViewModel, render_text


def launch_tk_window(ledger, version, invalid_stage_ids=None, snapshots=None):
    import tkinter as tk

    view = DiagnosticViewModel(ledger, version, invalid_stage_ids, snapshots).build()
    root = tk.Tk()
    root.title(f"应急预案演练诊断工作台 - {view['plan_id']} v{view['plan_version']}")

    header = tk.Label(root, text=f"预案版本 v{view['plan_version']}    统计 {view['stats']}", anchor="w")
    header.pack(fill="x", padx=8, pady=(8, 4))

    stage_frame = tk.LabelFrame(root, text="阶段依赖 / 状态 / 跳过原因")
    stage_frame.pack(fill="both", expand=True, padx=8, pady=4)
    stage_columns = ("order", "stage_id", "state", "prereq", "reason", "skip_trace", "flags", "version")
    stage_tree = _tree(stage_frame, stage_columns)
    for row in view["stage_rows"]:
        flags = " ".join(
            label for flag, label in ((row["unknown"], "UNKNOWN"), (row["stale"], "STALE")) if flag
        )
        stage_tree.insert(
            "",
            "end",
            values=(
                row["order"],
                row["stage_id"],
                row["state"],
                ",".join(row["prerequisites"]),
                row["reason"] or "",
                "->".join(row["skip_trace"]),
                flags,
                f"v{row['plan_version']}",
            ),
        )

    invalid_frame = tk.LabelFrame(root, text="本轮失效集合（稳定拓扑序）")
    invalid_frame.pack(fill="x", padx=8, pady=4)
    tk.Label(invalid_frame, text=", ".join(view["invalid_stage_ids"]) or "-", anchor="w").pack(fill="x")

    event_frame = tk.LabelFrame(root, text="注入事件时间轴")
    event_frame.pack(fill="both", expand=True, padx=8, pady=4)
    event_tree = _tree(event_frame, ("offset_ms", "event_id", "kind", "target_stage"))
    for event in view["event_timeline"]:
        event_tree.insert("", "end", values=(
            event["offset_ms"], event["event_id"], event["kind"], event["target_stage"]
        ))

    footer = tk.LabelFrame(root, text="版本 / 未知阶段 / 陈旧标记 / 快照")
    footer.pack(fill="x", padx=8, pady=(4, 8))
    tk.Label(
        footer,
        text=f"陈旧版本: {view['stale_versions']}    快照: {view['snapshots']}",
        anchor="w",
    ).pack(fill="x")

    def refresh():
        # 仅重新读取并渲染，不调用任何写接口，保证刷新不改变账本内容。
        refreshed = DiagnosticViewModel(ledger, version, invalid_stage_ids, snapshots).build()
        for item in stage_tree.get_children():
            stage_tree.delete(item)
        for row in refreshed["stage_rows"]:
            flags = " ".join(
                label for flag, label in ((row["unknown"], "UNKNOWN"), (row["stale"], "STALE")) if flag
            )
            stage_tree.insert(
                "",
                "end",
                values=(
                    row["order"], row["stage_id"], row["state"], ",".join(row["prerequisites"]),
                    row["reason"] or "", "->".join(row["skip_trace"]), flags, f"v{row['plan_version']}",
                ),
            )

    tk.Button(root, text="刷新（只读）", command=refresh).pack(pady=4)
    root.mainloop()
    return view


def _tree(parent, columns):
    import tkinter.ttk as ttk

    tree = ttk.Treeview(parent, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=110, anchor="w")
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return tree


def launch_or_text(ledger, version, invalid_stage_ids=None, snapshots=None):
    try:
        return launch_tk_window(ledger, version, invalid_stage_ids, snapshots)
    except Exception as exc:  # 无显示环境时降级为稳定文本视图。
        view = DiagnosticViewModel(ledger, version, invalid_stage_ids, snapshots).build()
        view["tk_unavailable_reason"] = repr(exc)
        return view


def render_for_terminal(ledger, version, invalid_stage_ids=None, snapshots=None):
    view = DiagnosticViewModel(ledger, version, invalid_stage_ids, snapshots).build()
    return render_text(view)
