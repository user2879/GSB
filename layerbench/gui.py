"""可选 Tkinter 诊断窗口：展示构建结果、缓存命中与预算占用。

仅依赖标准库 tkinter；在无显示/未带 Tkinter 的环境导入会抛异常，
由 CLI 捕获并给出明确提示，不影响命令行与测试流程。
"""
from __future__ import annotations

import json
import os
import threading

from .app import Workbench


def launch_gui(workdir: str, manifest_path: str,
               source_root: str | None = None) -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("layerbench 分层构建与缓存诊断工作台")
    root.geometry("900x600")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text=f"工作目录: {workdir}").pack(anchor="w")
    ttk.Label(top, text=f"清单: {manifest_path}").pack(anchor="w")
    status_var = tk.StringVar(value="就绪")
    ttk.Label(top, textvariable=status_var).pack(anchor="w")

    columns = ("stage", "seq", "type", "status", "code", "cache", "digest")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    headers = {
        "stage": "阶段", "seq": "序号", "type": "类型", "status": "状态",
        "code": "业务码", "cache": "缓存命中", "digest": "层摘要(前12位)",
    }
    for col in columns:
        tree.heading(col, text=headers[col])
        tree.column(col, width=110, anchor="w")
    tree.pack(fill="both", expand=True, padx=8, pady=8)

    def _run_build() -> None:
        status_var.set("构建中…")

        def worker() -> None:
            try:
                bench = Workbench(workdir)
                result = bench.build(manifest_path, source_root=source_root)
                root.after(0, lambda: _populate(result))
            except Exception as exc:  # 展示业务错误，不使窗口崩溃
                root.after(0, lambda: messagebox.showerror(
                    "构建失败", str(exc)))
                root.after(0, lambda: status_var.set("构建失败"))

        threading.Thread(target=worker, daemon=True).start()

    def _populate(result: dict) -> None:
        for item in tree.get_children():
            tree.delete(item)
        for node in result["report"]["nodes"]:
            tree.insert("", "end", values=(
                node["stage"], node["seq"], node["type"], node["status"],
                node["result_code"], "是" if node["cache_hit"] else "否",
                (node["layer_digest"] or "")[:12],
            ))
        peak = result["report"]["budget"]["peak_global_running"]
        status_var.set(
            f"完成：{result['build_id']} 业务码="
            f"{result['report']['build_code']} 峰值槽位={peak}"
        )

    ttk.Button(top, text="运行构建", command=_run_build).pack(anchor="e")
    root.mainloop()
