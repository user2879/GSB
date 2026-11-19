"""Tkinter 诊断界面。

Tkinter 缺失(如部分 Linux 最小安装)时 import 不报错, 仅在真正启动时给出提示;
Windows 官方 CPython 安装包默认包含 Tcl/Tk, 可直接 launch()。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_CAPACITY, DEFAULT_PAGE_SIZE
from .gui_model import collect_model

try:  # 防御性导入: 核心功能不依赖 Tk
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    _TK_AVAILABLE = True
except Exception:  # noqa: BLE001
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    _TK_AVAILABLE = False


class DiagnosticApp:  # pragma: no cover - 需图形环境
    def __init__(self, root: "tk.Tk", initial_dir: str | None = None) -> None:
        self.root = root
        self.root.title("页级双写缓冲恢复诊断")
        self.page_size = DEFAULT_PAGE_SIZE
        self.capacity = DEFAULT_CAPACITY
        self.dir_var = tk.StringVar(value=initial_dir or "")
        self._build()
        if initial_dir:
            self.refresh()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="会话目录:").pack(side="left")
        ttk.Entry(top, textvariable=self.dir_var, width=60).pack(side="left", padx=4)
        ttk.Button(top, text="浏览", command=self.choose_dir).pack(side="left")
        ttk.Button(top, text="刷新", command=self.refresh).pack(side="left", padx=4)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.tables: dict[str, ttk.Treeview] = {}
        for tab, columns in [
            ("buffer", ("offset", "page_no", "seq", "digest")),
            ("main", ("page_no", "digest", "expected", "intact")),
            ("markers", ("commit_id", "phase", "seq", "ts_ms")),
            ("codes", ("ts_ms", "event", "code")),
            ("snapshots", ("session_id", "created_seq", "buffer_digest")),
        ]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=tab)
            tree = ttk.Treeview(frame, columns=columns, show="headings")
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=160 if col != "digest" else 320)
            vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            tree.pack(side="left", fill="both", expand=True)
            vsb.pack(side="right", fill="y")
            self.tables[tab] = tree

    def choose_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.dir_var.set(path)
            self.refresh()

    def _clear(self, tab: str) -> None:
        self.tables[tab].delete(*self.tables[tab].get_children())

    def refresh(self) -> None:
        root_dir = self.dir_var.get().strip()
        if not root_dir or not Path(root_dir).exists():
            messagebox.showwarning("提示", "请选择有效的会话目录")
            return
        model = collect_model(root_dir, self.page_size, self.capacity)
        for tab in self.tables:
            self._clear(tab)
        for row in model["tabs"]["buffer"]["mirrors"]:
            self.tables["buffer"].insert(
                "", "end",
                values=(row["offset"], row["page_no"], row["seq"], row["digest"]),
            )
        for row in model["tabs"]["main"]["pages"]:
            self.tables["main"].insert(
                "", "end",
                values=(row["page_no"], row["digest"], row["expected"], row["intact"]),
            )
        for row in model["tabs"]["markers"]["markers"]:
            self.tables["markers"].insert(
                "", "end",
                values=(row["commit_id"], row["phase"], row["seq"], row["ts_ms"]),
            )
        for row in model["tabs"]["codes"]["events"]:
            self.tables["codes"].insert(
                "", "end",
                values=(row["ts_ms"], row["event"], row["code"]),
            )
        for row in model["tabs"]["snapshots"]["snapshots"]:
            self.tables["snapshots"].insert(
                "", "end",
                values=(row["session_id"], row["created_seq"], row["buffer_digest"]),
            )
        err = model["tabs"]["buffer"].get("error")
        if err:
            messagebox.showerror("缓冲校验", str(err))


def launch(initial_dir: str | None = None) -> int:  # pragma: no cover
    if not _TK_AVAILABLE:
        raise RuntimeError(
            "当前解释器缺少 tkinter, 请安装带 Tcl/Tk 的 CPython(Windows 安装包勾选 Tcl/Tk)"
        )
    root = tk.Tk()
    root.geometry("980x640")
    DiagnosticApp(root, initial_dir)
    root.mainloop()
    return 0
