"""Tkinter 演示界面（可选依赖：无 Tkinter 时给出明确启动指引）。

界面在后台线程跑会话引擎，主线程保持 Tk 事件循环；所有输出汇总到只读
文本区，错误以业务码形式展示。
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict

from .cli import (
    DEFAULT_DB,
    DEFAULT_LOG,
    DEFAULT_MANIFEST,
    DEFAULT_NODES,
    WORKSPACE,
)

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
except ImportError:  # pragma: no cover - 取决于系统 Tk 安装
    tk = None


class EngineGUI:  # pragma: no cover - 需要图形环境
    def __init__(self, root: "tk.Tk"):
        self.root = root
        root.title("跨组件配置分发与一致性审计平台")
        root.geometry("900x640")

        frm = ttk.Frame(root, padding=8)
        frm.pack(fill=tk.X)

        ttk.Label(frm, text="缩放指数 exponent:").grid(row=0, column=0, sticky=tk.W)
        self.exponent = tk.IntVar(value=3)
        ttk.Spinbox(frm, from_=0, to=9, textvariable=self.exponent, width=5).grid(
            row=0, column=1, padx=4
        )
        ttk.Label(frm, text="并发上限:").grid(row=0, column=2, sticky=tk.W)
        self.concurrency = tk.IntVar(value=2)
        ttk.Spinbox(frm, from_=1, to=8, textvariable=self.concurrency, width=5).grid(
            row=0, column=3, padx=4
        )

        ttk.Button(frm, text="清空运行时", command=self.clean).grid(row=0, column=4, padx=4)
        ttk.Button(frm, text="运行全版本", command=self.run_all).grid(row=0, column=5, padx=4)
        ttk.Button(frm, text="恢复快照", command=self.restore).grid(row=0, column=6, padx=4)
        ttk.Button(frm, text="重放校验", command=self.replay).grid(row=0, column=7, padx=4)

        self.text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
        self.text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._write_line(f"工作区: {WORKSPACE}")

    def _write_line(self, content: str) -> None:
        self.text.insert(tk.END, content + "\n")
        self.text.see(tk.END)

    def _run_async(self, fn) -> None:
        def worker() -> None:
            try:
                result = fn()
                payload = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as exc:  # 界面里集中展示业务码
                payload = f"ERROR: {exc}"
            self.root.after(0, lambda: self._write_line(payload))

        threading.Thread(target=worker, daemon=True).start()

    def _engine_kwargs(self) -> Dict[str, Any]:
        return {
            "workdir": WORKSPACE,
            "manifest_path": DEFAULT_MANIFEST,
            "nodes_path": DEFAULT_NODES,
            "db_path": DEFAULT_DB,
            "log_path": DEFAULT_LOG,
            "exponent": self.exponent.get(),
            "max_concurrency": self.concurrency.get(),
        }

    def clean(self) -> None:
        import shutil

        runtime = WORKSPACE / "runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir(parents=True, exist_ok=True)
        self._write_line("已清空 runtime/")

    def run_all(self) -> None:
        from .engine import SessionEngine

        def job() -> Dict[str, Any]:
            engine = SessionEngine(**self._engine_kwargs())
            try:
                return {"summaries": engine.process_versions()}
            finally:
                engine.close()

        self._run_async(job)

    def restore(self) -> None:
        from .engine import SessionEngine

        def job() -> Dict[str, Any]:
            engine = SessionEngine(**self._engine_kwargs())
            try:
                info = engine.restore_latest()
                info.pop("snapshot", None)
                return info
            finally:
                engine.close()

        self._run_async(job)

    def replay(self) -> None:
        from .engine import SessionEngine

        def job() -> Dict[str, Any]:
            engine = SessionEngine(**self._engine_kwargs())
            try:
                engine.restore_latest()
                return engine.replay()
            finally:
                engine.close()

        self._run_async(job)


def main() -> int:
    if tk is None:
        print(
            "当前 Python 缺少 Tkinter。\n"
            "Windows 官方安装包默认自带 Tkinter；若缺失请在安装器勾选 "
            "'tcl/tk and IDLE' 后重装 Python。\n"
            "Debian/Ubuntu 环境可执行: sudo apt install python3-tk\n"
            "无图形环境时可使用命令行: python -m ccdplatform.cli run"
        )
        return 2
    root = tk.Tk()
    EngineGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
