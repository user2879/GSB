"""可选 Tkinter 汇总窗口（惰性导入，缺 Tk 时由 CLI 自动降级为终端输出）。"""
from __future__ import annotations

import json

import tkinter as tk
from tkinter import ttk


def show_summary(payload: dict) -> None:
    root = tk.Tk()
    root.title("降级传播与故障域收敛诊断")
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")

    text = tk.Text(frame, width=96, height=32, wrap="word")
    text.grid(row=0, column=0, sticky="nsew")
    text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))
    text.configure(state="disabled")

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    text.configure(yscrollcommand=scrollbar.set)
    root.mainloop()
