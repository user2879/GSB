"""Tkinter 诊断工作台。

Tkinter 在模块内惰性导入：无显示环境或解释器缺少 tkinter 时，
核心数据视图模型仍可通过 workspace 模块无头测试，GUI 仅在显式启动时依赖 Tk。
"""

from pathlib import Path

from .samples import write_samples
from .spec import load_manifest
from .workspace import TABLE_COLUMNS, build_table_rows, heat_level, list_snapshots
from .analytics import queue_pressure_series


class HeadlessUnavailable(RuntimeError):
    pass


def launch(workspace_dir):
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise HeadlessUnavailable(
            "当前 Python 解释器缺少 Tkinter；Windows 标准安装 python.org 版本自带 Tkinter。"
        ) from exc
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        raise HeadlessUnavailable(f"无图形显示环境，无法启动 Tk 窗口: {exc}") from exc

    workspace = Path(workspace_dir)
    sample_dir = workspace.parent / "samples"
    write_samples(sample_dir)
    manifest = load_manifest(sample_dir / "manifest_v1.json")
    from .engine import Engine

    engine = Engine(manifest, manifest["functions"][0]["function_id"],
                    manifest["functions"][0]["function_version"])
    payloads = [r.to_dict() for r in engine.run_all(manifest["call_sequence"])]

    root.title("Serverless 冷启动与并发控制诊断台")
    header = ttk.Label(
        root,
        text=f"当前声明版本 v{manifest['declaration_version']}  缩放因子 {manifest['scale']}  "
             f"（界面每个数字均绑定该声明版本）",
    )
    header.pack(anchor="w", padx=8, pady=6)

    cols = tuple(c[0] for c in TABLE_COLUMNS)
    tree = ttk.Treeview(root, columns=cols, show="headings", height=14)
    for key, label in TABLE_COLUMNS:
        tree.heading(key, text=label)
        tree.column(key, width=110, anchor="center")
    for row in build_table_rows(payloads):
        tree.insert("", "end", values=tuple(row[c] for c in cols))
    tree.pack(fill="x", padx=8)

    heat = ttk.Frame(root)
    heat.pack(fill="x", padx=8, pady=6)
    ttk.Label(heat, text="冷启动比例热力:").pack(anchor="w")
    for p in payloads:
        level, name = heat_level(p["cold_ratio_q"], manifest["scale"])
        color = {0: "#2e7d32", 1: "#f9a825", 2: "#ef6c00", 3: "#c62828"}[level]
        cell = tk.Label(heat, text=f"m{p['minute']}:{name}", bg=color, fg="white", width=8)
        cell.pack(side="left", padx=1)

    curve = tk.Canvas(root, height=140, bg="white")
    curve.pack(fill="x", padx=8, pady=6)
    series = queue_pressure_series(payloads)
    max_wait = max((s["avg_wait_ms"] for s in series), default=1) or 1
    width = max(640, len(series) * 48)
    curve.config(width=width)
    prev_x, prev_y = None, None
    for i, s in enumerate(series):
        x = 30 + i * 48
        y = 120 - int(100 * s["avg_wait_ms"] / max_wait)
        curve.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#1565c0")
        curve.create_text(x, 132, text=f"m{s['minute']}", font=("TkDefaultFont", 7))
        if prev_x is not None:
            curve.create_line(prev_x, prev_y, x, y, fill="#1565c0")
        prev_x, prev_y = x, y
    curve.create_text(8, 8, anchor="w", text="平均排队等待 (ms)", font=("TkDefaultFont", 8))

    snap_frame = ttk.Frame(root)
    snap_frame.pack(fill="x", padx=8, pady=6)
    ttk.Label(snap_frame, text="快照列表（稳定顺序）:").pack(anchor="w")
    for snap in list_snapshots(workspace / "snapshots"):
        ttk.Label(
            snap_frame,
            text=f"{snap['run_id']}  v{snap['declaration_version']}  "
                 f"scale={snap['scale']}  minutes={snap['minutes']}",
        ).pack(anchor="w")

    root.mainloop()
    return root


def main():
    workspace = Path(__file__).resolve().parent.parent / "workspace"
    try:
        launch(workspace)
    except HeadlessUnavailable as exc:
        print(f"[GUI 不可用] {exc}")
        print("可在带桌面的 Windows 上执行: python -m coldstart_diag.gui")


if __name__ == "__main__":
    main()
