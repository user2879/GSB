from __future__ import annotations

import json
from pathlib import Path
import threading

from .cli import _demo
from .loader import load_workload
from .session import ExperimentSession


def launch(workload_path: str, db_path: str) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("JoinLab - 本地连接顺序与代价实验台")
    root.geometry("1100x760")

    workload_var = tk.StringVar(value=workload_path)
    db_var = tk.StringVar(value=db_path)
    output = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))

    def append(text: str) -> None:
        output.insert(tk.END, text + "\n")
        output.see(tk.END)

    def choose_workload() -> None:
        selected = filedialog.askopenfilename(
            title="选择工作负载 JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if selected:
            workload_var.set(selected)

    def choose_db() -> None:
        selected = filedialog.asksaveasfilename(
            title="选择 SQLite 账本",
            defaultextension=".sqlite",
            filetypes=[("SQLite", "*.sqlite *.db"), ("All files", "*.*")],
        )
        if selected:
            db_var.set(selected)

    def analyze() -> None:
        try:
            session = ExperimentSession(load_workload(workload_var.get()))
            append(json.dumps(session.analysis_report(), ensure_ascii=False, indent=2))
        except BaseException as exc:
            messagebox.showerror("分析失败", str(exc))

    def run_demo() -> None:
        def finish_success(report_path: str) -> None:
            append(f"演示完成，报告：{report_path}")

        def finish_failure(exc: BaseException) -> None:
            append(f"运行失败: {type(exc).__name__}: {exc}")
            messagebox.showerror("运行失败", str(exc))

        def worker() -> None:
            try:
                args = type(
                    "Args",
                    (),
                    {
                        "workload": workload_var.get(),
                        "db": db_var.get(),
                        "out": str(Path(db_var.get()).with_suffix(".report.json")),
                        "reset": False,
                        "interrupt_after_execution": None,
                        "interrupt_before_commit": None,
                        "fail_baseline": False,
                        "tamper_ledger": None,
                        "no_reexecute_replay": False,
                        "changes": None,
                    },
                )()
                code = _demo(args)
                root.after(0, lambda: finish_success(args.out))
            except BaseException as exc:
                root.after(0, lambda error=exc: finish_failure(error))

        threading.Thread(target=worker, daemon=True).start()

    toolbar = tk.Frame(root)
    toolbar.pack(fill=tk.X, padx=10, pady=8)
    tk.Label(toolbar, text="工作负载").pack(side=tk.LEFT)
    tk.Entry(toolbar, textvariable=workload_var, width=55).pack(side=tk.LEFT, padx=6)
    tk.Button(toolbar, text="浏览", command=choose_workload).pack(side=tk.LEFT)
    tk.Label(toolbar, text="账本").pack(side=tk.LEFT, padx=(12, 0))
    tk.Entry(toolbar, textvariable=db_var, width=35).pack(side=tk.LEFT, padx=6)
    tk.Button(toolbar, text="浏览", command=choose_db).pack(side=tk.LEFT)
    buttons = tk.Frame(root)
    buttons.pack(fill=tk.X, padx=10)
    tk.Button(buttons, text="加载与规范化分析", command=analyze).pack(side=tk.LEFT)
    tk.Button(buttons, text="完整演示（执行 / 账本 / 快照 / 重放）", command=run_demo).pack(
        side=tk.LEFT, padx=8
    )
    output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    append("选择本地 JSON 后点击“完整演示”。所有计算均在本机完成。")
    root.mainloop()
