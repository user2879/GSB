"""Tkinter 桌面工作台。

Windows 官方 Python 安装包自带 tkinter，可直接运行；在无显示或未安装
tkinter 的环境（如最小化 Linux 容器）中 launch_gui 返回 False 并给出提示，
CLI 与库 API 仍然完整可用。
"""
from __future__ import annotations

import json
import os
import traceback

from .errors import RepairError


def launch_gui(work_root: str = "demo_workspace") -> bool:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception:
        print("[gui] tkinter 不可用，请使用 CLI: python -m voice_repair_workbench demo")
        return False

    from .cli import demo, generate

    root = tk.Tk()
    root.title("语音录音修复工作台")
    root.geometry("860x600")

    ttk.Label(root, text="工作区目录:").pack(anchor="w", padx=10, pady=(10, 0))
    work_var = tk.StringVar(value=work_root)
    ttk.Entry(root, textvariable=work_var, width=90).pack(padx=10, fill="x")

    text = tk.Text(root, wrap="word", height=30)
    text.pack(padx=10, pady=10, fill="both", expand=True)

    def log(line: str) -> None:
        text.insert("end", line + "\n")
        text.see("end")
        root.update_idletasks()

    def run_generate() -> None:
        try:
            paths = generate(os.path.join(work_var.get(), "materials"))
            log(json.dumps(paths, ensure_ascii=False, indent=2))
        except RepairError as exc:
            messagebox.showerror(exc.code, json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        except Exception:
            messagebox.showerror("unexpected", traceback.format_exc())

    def run_demo() -> None:
        try:
            result = demo(work_var.get())
            log("演示完成，步骤数: " + str(len(result["steps"])))
            log("报告: " + result["segment_metrics_report"])
        except RepairError as exc:
            messagebox.showerror(exc.code, json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
        except Exception:
            messagebox.showerror("unexpected", traceback.format_exc())

    bar = ttk.Frame(root)
    bar.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Button(bar, text="生成确定性素材", command=run_generate).pack(side="left")
    ttk.Button(bar, text="运行完整修复演示", command=run_demo).pack(side="left", padx=8)
    ttk.Button(bar, text="退出", command=root.destroy).pack(side="right")

    root.mainloop()
    return True


if __name__ == "__main__":
    launch_gui()
