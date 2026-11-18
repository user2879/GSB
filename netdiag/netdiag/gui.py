"""可选 Tkinter 图形外壳（Windows 标准库自带 Tkinter）。

无显示环境或缺少 Tkinter 时导入不报错，仅在真正启动 GUI 时抛出明确提示。
"""

from __future__ import annotations

from typing import Any, Dict


def launch() -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "Tkinter 不可用；无头环境请使用 CLI（python -m netdiag.cli）。"
            f"原始错误: {exc}") from exc

    from .pipeline import run_diagnosis
    from .logging_utils import StructuredLogger

    root = tk.Tk()
    root.title("丢包与网络故障诊断工作台")

    status = tk.StringVar(value="填写样本路径后运行诊断")
    wire_var = tk.StringVar(value="output/sample/capture.bin")
    event_var = tk.StringVar(
        value="output/sample/events_v1.json,output/sample/events_v2.json,"
              "output/sample/events_v3.json")
    window_var = tk.StringVar(value="100")

    def run() -> None:
        import json
        events = [p.strip() for p in event_var.get().split(",") if p.strip()]
        logger = StructuredLogger(stream=None)
        try:
            session = run_diagnosis(
                wire_paths=[wire_var.get()], event_paths=events,
                window_ms=int(window_var.get()), logger=logger)
            text.delete("1.0", tk.END)
            text.insert(tk.END, json.dumps(
                {f: c.to_dict() for f, c in
                 session.conclusions.items()}, indent=2, ensure_ascii=False))
            status.set(f"诊断完成：{len(session.model.flows)} 条流")
        except Exception as exc:  # pragma: no cover - GUI 错误展示
            status.set(f"诊断失败: {exc}")

    frame = ttk.Frame(root, padding=8)
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(frame, text="抓包样本:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frame, textvariable=wire_var, width=60).grid(row=0, column=1)
    ttk.Label(frame, text="事件清单:").grid(row=1, column=0, sticky="w")
    ttk.Entry(frame, textvariable=event_var, width=60).grid(row=1, column=1)
    ttk.Label(frame, text="窗口(ms):").grid(row=2, column=0, sticky="w")
    ttk.Entry(frame, textvariable=window_var, width=10).grid(row=2, column=1,
                                                              sticky="w")
    ttk.Button(frame, text="运行诊断", command=run).grid(row=3, column=1,
                                                          sticky="w", pady=6)
    text = tk.Text(root, width=100, height=30)
    text.grid(row=1, column=0, padx=8, pady=8)
    ttk.Label(root, textvariable=status).grid(row=2, column=0, sticky="w",
                                               padx=8, pady=4)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    launch()
