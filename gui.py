#!/usr/bin/env python3
"""Tkinter 账本查看器.

展示记录列表与格式版本、升级轨迹、补齐字段来源、一致性结论与快照列表;
渲染顺序稳定(按到达序号/请求标识排序).
"""

import json
import os
import sys


def main(ws):
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter 不可用: 请使用 Windows 官方 Python 安装包(默认包含 tkinter).",
              file=sys.stderr)
        return 1
    rt = os.path.join(ws, "runtime")
    report_path = os.path.join(rt, "run_report.json")
    report = {}
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    root = tk.Tk()
    root.title("推理请求归属账本互操作工具")
    root.geometry("980x640")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    frame_records = ttk.Frame(notebook)
    notebook.add(frame_records, text="记录列表")
    cols = ("request_id", "fmt", "batch_slot", "arrival_seq", "group_rank", "filled")
    tree = ttk.Treeview(frame_records, columns=cols, show="headings")
    for col, label in zip(cols, ("请求标识", "格式版本", "批槽位", "到达序号",
                                 "组内名次", "补齐字段来源")):
        tree.heading(col, text=label)
        tree.column(col, width=140)
    for rec in sorted(report.get("records", []),
                      key=lambda r: (r["arrival_seq"], r["request_id"])):
        filled = ",".join(f"{k}={v}" for k, v in sorted(rec["filled"].items()))
        tree.insert("", "end", values=(rec["request_id"], rec["fmt"],
                                       rec["batch_slot"], rec["arrival_seq"],
                                       rec["group_rank"], filled))
    tree.pack(fill="both", expand=True)
    frame_trail = ttk.Frame(notebook)
    notebook.add(frame_trail, text="升级轨迹与一致性")
    text = tk.Text(frame_trail, wrap="none")
    lines = ["== 升级轨迹 =="]
    for item in report.get("upgrade_trail", []):
        lines.append(f"{item['request_id']}: {item['from']} -> {item['to']} "
                     f"(cursor={item['cursor']})")
    lines.append("")
    lines.append("== 升级事件 ==")
    for ev in report.get("upgrade_events", []):
        lines.append(json.dumps(ev, sort_keys=True, ensure_ascii=False))
    lines.append("")
    lines.append("== 一致性结论 ==")
    lines.append(json.dumps(report.get("consistency", {}),
                            sort_keys=True, ensure_ascii=False, indent=2))
    text.insert("1.0", "\n".join(lines))
    text.configure(state="disabled")
    text.pack(fill="both", expand=True)
    frame_snaps = ttk.Frame(notebook)
    notebook.add(frame_snaps, text="快照列表")
    snap_list = tk.Listbox(frame_snaps)
    for path in sorted(report.get("snapshots", [])):
        snap_list.insert("end", os.path.basename(path))
    snap_list.pack(fill="both", expand=True)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(os.getcwd()))
