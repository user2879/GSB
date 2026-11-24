from __future__ import annotations

import json
import queue
import threading
from typing import Any

from .engine import ArbiterEngine
from .errors import AppError
from .ledger import Ledger
from .spec import load_spec


def launch_gui(spec_path: str, ledger_path: str) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise AppError(
            "E_TK",
            "gui.tkinter",
            "Tkinter is unavailable; use the CLI or install the official Python Tk component",
        ) from exc

    class App:
        def __init__(self, root: Any) -> None:
            self.root = root
            self.root.title("配置权威仲裁诊断")
            self.root.geometry("1280x780")
            self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
            self.engine: ArbiterEngine | None = None
            self.ledger: Ledger | None = None
            self.worker: threading.Thread | None = None
            self.spec_var = tk.StringVar(value=spec_path)
            self.ledger_var = tk.StringVar(value=ledger_path)
            self.status_var = tk.StringVar(value="Ready")
            self._build_widgets()
            self.root.after(100, self._poll_events)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        def _build_widgets(self) -> None:
            toolbar = ttk.Frame(self.root, padding=8)
            toolbar.pack(fill="x")
            ttk.Label(toolbar, text="Spec").pack(side="left")
            ttk.Entry(toolbar, textvariable=self.spec_var, width=42).pack(side="left", padx=4)
            ttk.Label(toolbar, text="Ledger").pack(side="left")
            ttk.Entry(toolbar, textvariable=self.ledger_var, width=34).pack(side="left", padx=4)
            ttk.Button(toolbar, text="加载", command=self.load).pack(side="left", padx=2)
            ttk.Button(toolbar, text="启动服务", command=self.start_services).pack(side="left", padx=2)
            ttk.Button(toolbar, text="执行仲裁", command=self.run_all).pack(side="left", padx=2)
            ttk.Button(toolbar, text="重放校验", command=self.replay).pack(side="left", padx=2)
            ttk.Button(toolbar, text="模拟切片变更", command=self.open_slice_dialog).pack(side="left", padx=2)
            ttk.Button(toolbar, text="停止服务", command=self.stop_services).pack(side="left", padx=2)

            paned = ttk.Panedwindow(self.root, orient="vertical")
            paned.pack(fill="both", expand=True, padx=8, pady=4)
            top = ttk.Frame(paned)
            bottom = ttk.Frame(paned)
            paned.add(top, weight=3)
            paned.add(bottom, weight=2)

            columns = ("view", "version", "key", "mode", "value", "adopted", "bound", "cause")
            self.tree = ttk.Treeview(top, columns=columns, show="headings")
            widths = {
                "view": 150,
                "version": 65,
                "key": 130,
                "mode": 145,
                "value": 100,
                "adopted": 190,
                "bound": 80,
                "cause": 190,
            }
            for column in columns:
                self.tree.heading(column, text=column)
                self.tree.column(column, width=widths[column], anchor="w")
            self.tree.pack(side="left", fill="both", expand=True)
            scrollbar = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
            scrollbar.pack(side="right", fill="y")
            self.tree.configure(yscrollcommand=scrollbar.set)
            self.tree.bind("<<TreeviewSelect>>", self._show_selected)

            self.detail = tk.Text(bottom, wrap="none", font=("Consolas", 9))
            self.detail.pack(fill="both", expand=True)
            status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
            status.pack(fill="x", side="bottom")
            self.row_payloads: dict[str, dict[str, Any]] = {}

        def _set_status(self, text: str) -> None:
            self.status_var.set(text)

        def _background(self, label: str, action: Any) -> None:
            if self.worker is not None and self.worker.is_alive():
                self._set_status("已有任务运行中")
                return
            self._set_status(label)

            def run() -> None:
                try:
                    result = action()
                    self.events.put(("ok", result))
                except Exception as exc:  # Reported through the UI queue.
                    self.events.put(("error", exc))

            self.worker = threading.Thread(target=run, name="gui-worker", daemon=True)
            self.worker.start()

        def _poll_events(self) -> None:
            try:
                while True:
                    kind, payload = self.events.get_nowait()
                    if kind == "error":
                        from tkinter import messagebox

                        if isinstance(payload, AppError):
                            messagebox.showerror("诊断失败", json.dumps(payload.to_dict(), ensure_ascii=False))
                            self._set_status(f"{payload.code}: {payload.message}")
                        else:
                            messagebox.showerror("诊断失败", str(payload))
                            self._set_status(str(payload))
                    else:
                        self._set_status("完成")
            except queue.Empty:
                pass
            self.root.after(100, self._poll_events)

        def load(self) -> None:
            def action() -> dict[str, Any]:
                self._close_engine()
                spec = load_spec(self.spec_var.get())
                ledger = Ledger(self.ledger_var.get())
                engine = ArbiterEngine(spec, ledger)
                self.engine = engine
                self.ledger = ledger
                return {
                    "format_version": spec.format_version,
                    "environment_count": len(spec.environments),
                    "view_count": len(spec.views),
                    "ledger": ledger.snapshot_report(),
                }

            self._background("加载声明", action)

        def start_services(self) -> None:
            if self.engine is None:
                self._set_status("请先加载声明")
                return
            engine = self.engine

            def action() -> dict[str, Any]:
                return {"start_errors": engine.start_services()}

            self._background("启动子进程服务", action)

        def run_all(self) -> None:
            if self.engine is None:
                self._set_status("请先加载声明")
                return
            engine = self.engine

            def action() -> dict[str, Any]:
                engine.start_services()
                report = engine.run_all()
                replay = engine.replay_current()
                self.root.after(0, lambda: self._render(report.to_dict()))
                return {"replay": replay, "snapshot": engine.ledger.snapshot_report()}

            self._background("执行仲裁并写入账本", action)

        def replay(self) -> None:
            if self.engine is None:
                self._set_status("请先加载声明")
                return
            engine = self.engine

            def action() -> dict[str, Any]:
                engine.start_services()
                return engine.replay_current()

            self._background("重放校验", action)

        def open_slice_dialog(self) -> None:
            if self.engine is None:
                self._set_status("请先加载声明")
                return
            from tkinter import messagebox

            dialog = tk.Toplevel(self.root)
            dialog.title("模拟环境切片变更")
            dialog.geometry("560x420")
            env_ids = sorted(self.engine.current_slices)
            env_var = tk.StringVar(value=env_ids[0])
            version_var = tk.StringVar(
                value=str(self.engine.current_slices[env_ids[0]].version + 1)
            )
            ttk.Label(dialog, text="环境").pack(anchor="w", padx=8, pady=4)
            ttk.Combobox(dialog, values=env_ids, textvariable=env_var, state="readonly").pack(
                fill="x", padx=8
            )
            ttk.Label(dialog, text="新切片版本").pack(anchor="w", padx=8, pady=4)
            ttk.Entry(dialog, textvariable=version_var).pack(fill="x", padx=8)
            ttk.Label(dialog, text="值 JSON").pack(anchor="w", padx=8, pady=4)
            text = tk.Text(dialog, font=("Consolas", 10))
            text.pack(fill="both", expand=True, padx=8, pady=4)
            current = self.engine.current_slices[env_ids[0]]
            text.insert("1.0", json.dumps(dict(current.values), ensure_ascii=False, indent=2))

            def apply_change() -> None:
                try:
                    version = int(version_var.get())
                    values = json.loads(text.get("1.0", "end"))
                    if not isinstance(values, dict):
                        raise ValueError("values must be a JSON object")
                    env_id = env_var.get()
                except Exception as exc:
                    messagebox.showerror("输入错误", str(exc))
                    return
                engine = self.engine
                assert engine is not None

                def action() -> dict[str, Any]:
                    reports = engine.change_environment_slice(
                        env_id, version=version, values=values
                    )
                    self.root.after(0, lambda: self._render(engine.current_display()))
                    return {
                        "invalidations": [item.to_dict() for item in reports],
                        "replay": engine.replay_current(),
                    }

                dialog.destroy()
                self._background("切片变更与失效传播", action)

            ttk.Button(dialog, text="应用", command=apply_change).pack(pady=8)

        def _render(self, payload: dict[str, Any]) -> None:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.row_payloads.clear()
            if "views" in payload and isinstance(payload["views"], list):
                for run in payload["views"]:
                    summary = run.get("summary", {})
                    for key in summary.get("keys", []):
                        self._insert_row(summary.get("view_id"), summary.get("version"), key)
            elif "views" in payload and isinstance(payload["views"], dict):
                for view_id in sorted(payload["views"]):
                    for key in payload["views"][view_id]:
                        self._insert_row(view_id, key.get("view_version"), key, current=True)
            self.detail.delete("1.0", "end")
            self.detail.insert("end", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

        def _insert_row(
            self,
            view_id: Any,
            version: Any,
            key: dict[str, Any],
            current: bool = False,
        ) -> None:
            summary = key.get("summary", {}) if isinstance(key.get("summary"), dict) else {}
            value = summary.get("decimal", summary.get("units"))
            bound = key.get("error_bound", {}).get("decision_bound_units")
            cause = key.get("attribution", {}).get("primary_cause")
            row_id = self.tree.insert(
                "",
                "end",
                values=(
                    view_id,
                    version,
                    key.get("key"),
                    key.get("mode"),
                    value,
                    ",".join(key.get("adopted", [])),
                    bound,
                    cause,
                ),
            )
            self.row_payloads[row_id] = key

        def _show_selected(self, event: Any) -> None:
            selection = self.tree.selection()
            if not selection:
                return
            payload = self.row_payloads.get(selection[0])
            if payload is None:
                return
            self.detail.delete("1.0", "end")
            self.detail.insert("end", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

        def stop_services(self) -> None:
            if self.engine is not None:
                self.engine.stop_services()
            self._set_status("服务已停止")

        def _close_engine(self) -> None:
            if self.engine is not None:
                self.engine.close()
            elif self.ledger is not None:
                self.ledger.close()
            self.engine = None
            self.ledger = None

        def _on_close(self) -> None:
            self._close_engine()
            self.root.destroy()

    try:
        root = tk.Tk()
    except Exception as exc:
        raise AppError("E_TK", "gui.display", f"cannot open Tk window: {exc}") from exc
    App(root)
    root.mainloop()
