"""可选 Tkinter 交互界面：前缀键集合/计数快速查询 + 权威校验。

Windows 官方 Python 安装包自带 tkinter，可直接 ``python -m kvstore.gui``。
在无显示环境（如无 X 的 WSL/服务器）上导入本模块会在创建窗口时得到 TclError，
此时请使用 CLI（``python -m kvstore.cli``），引擎能力完全相同。
"""

from __future__ import annotations

import json
from typing import Any

from .engine import Engine
from .errors import KvError
from .replay import verify_visible
from .storage import Storage
from .trace import load_trace


class KvStoreGui:
    def __init__(self, db_path: str, trace_path: str | None = None,
                 initial_path: str | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk, messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.storage = Storage(db_path)
        self.engine = Engine(self.storage)
        self.trace_path = trace_path
        self.initial_path = initial_path

        self.root = tk.Tk()
        self.root.title("嵌入式键值索引与过期组件")
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="前缀").grid(row=0, column=0, sticky="w")
        self.prefix_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.prefix_var, width=32).grid(row=0, column=1)

        ttk.Label(frame, text="逻辑时钟").grid(row=1, column=0, sticky="w")
        self.clock_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.clock_var, width=32).grid(row=1, column=1)

        ttk.Button(frame, text="快速查询", command=self.do_query).grid(row=2, column=0, pady=6)
        ttk.Button(frame, text="权威重放校验", command=self.do_verify).grid(row=2, column=1)

        self.output = tk.Text(frame, width=72, height=24)
        self.output.grid(row=3, column=0, columnspan=2)

    def do_query(self) -> None:
        try:
            clock = int(self.clock_var.get())
            result = self.engine.query_prefix(self.prefix_var.get(), clock)
            self._show(result)
        except (KvError, ValueError) as exc:
            self.messagebox.showerror("查询失败", str(exc))

    def do_verify(self) -> None:
        if not self.trace_path:
            self.messagebox.showwarning("缺少轨迹", "启动时未提供 --trace，无法权威重放")
            return
        try:
            clock = int(self.clock_var.get())
            operations = load_trace(self.trace_path)
            initial = []
            if self.initial_path:
                with open(self.initial_path, "r", encoding="utf-8") as handle:
                    initial = json.load(handle)
            result = verify_visible(
                self.engine, operations, initial,
                self.storage.db_path + ".gui_replay",
                clock=clock, prefixes=[self.prefix_var.get()],
            )
            self._show(result)
        except KvError as exc:
            self._show(exc.to_dict())
        except ValueError as exc:
            self.messagebox.showerror("校验失败", str(exc))

    def _show(self, payload: Any) -> None:
        self.output.delete("1.0", self.tk.END)
        self.output.insert(self.tk.END, json.dumps(payload, ensure_ascii=False,
                                                    sort_keys=True, indent=2))

    def run(self) -> None:
        self.root.mainloop()
        self.storage.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="键值索引交互 GUI")
    parser.add_argument("--db", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--initial")
    args = parser.parse_args()
    KvStoreGui(args.db, args.trace, args.initial).run()


if __name__ == "__main__":
    main()
