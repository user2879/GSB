"""Tkinter 诊断界面：栏路径、预算消耗、待裁决栏、失效/复用条目、参数增量重算。"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config, write_default_config
from .pipeline import Pipeline


def launch_gui(root: Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except Exception as exc:  # 无显示或未装 python3-tk
        raise RuntimeError(
            "Tkinter 不可用（可执行 sudo apt install python3-tk 后重试）；"
            "核心引擎与测试不依赖界面。原始错误: " + str(exc)) from exc

    config_path = root / "config" / "config.json"
    pipeline = Pipeline(root / "samples", Config.load(config_path),
                        root / "snapshots")

    app = tk.Tk()
    app.title("竖排古籍版面识别诊断台")
    app.geometry("980x720")

    editor = ttk.Frame(app)
    editor.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
    fields = {}
    for idx, key in enumerate(("window_size", "window_stride", "scan_row_budget",
                               "compare_point_budget", "adjudication_budget",
                               "max_workers")):
        ttk.Label(editor, text=key).grid(row=0, column=idx * 2, padx=2)
        var = tk.StringVar(value=str(getattr(pipeline.cfg, key)))
        ttk.Entry(editor, textvariable=var, width=8).grid(row=0, column=idx * 2 + 1)
        fields[key] = var

    text = tk.Text(app, wrap=tk.NONE)
    text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def refresh() -> None:
        state = pipeline.execute()
        text.delete("1.0", tk.END)
        text.insert(tk.END, json.dumps(state, ensure_ascii=False, indent=2))

    def apply_and_run() -> None:
        try:
            payload = dict(Config.load(config_path).to_dict())
            for key, var in fields.items():
                payload[key] = int(var.get())
            cfg = Config.from_dict(payload)
            write_default_config(config_path, cfg.to_dict())
            pipeline.cfg = cfg
            pipeline.scheduler = type(pipeline.scheduler)(cfg)
            refresh()
        except Exception as exc:
            messagebox.showerror("配置不合规", str(exc))

    ttk.Button(editor, text="应用参数并增量重算", command=apply_and_run).grid(
        row=0, column=12, padx=6)
    ttk.Button(editor, text="仅刷新", command=refresh).grid(row=0, column=13, padx=6)
    refresh()
    app.mainloop()
