"""Tkinter 专业桌面界面（懒加载 tkinter）。

后台线程执行迁移/恢复等耗时动作，通过队列向主线程推送结构化日志，
保证界面不卡死；所有业务逻辑仍复用核心模块，GUI 只做编排与展示。
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from .applog import StructuredLogger
from .engine import TierEngine
from .errors import TierMoverError
from .ledger import Ledger
from .manifest import load_manifest
from .planner import derive_plan
from .recovery import RecoveryService
from .replay import ReplayService
from .workspace import bootstrap_workspace, workspace_paths


class TierMoverApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("冷热分层与迁移组件")
        self.root.geometry("1080x760")
        self.events: queue.Queue[dict] = queue.Queue()
        self.workspace: Path | None = None
        self.engine: TierEngine | None = None
        self.ledger: Ledger | None = None
        self.manifest = None
        self.logger: StructuredLogger | None = None
        self._build_layout()
        self._poll_events()

    def _build_layout(self) -> None:
        ttk = self.ttk
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="工作区目录:").pack(side="left")
        self.ws_var = self.tk.StringVar(value=str(Path.cwd() / "tier_workspace"))
        ttk.Entry(top, textvariable=self.ws_var, width=70).pack(
            side="left", padx=6)
        ttk.Button(top, text="初始化工作区",
                   command=self.init_workspace).pack(side="left", padx=3)
        ttk.Button(top, text="加载/校验",
                   command=self.load_workspace).pack(side="left", padx=3)

        actions = ttk.Frame(self.root, padding=(8, 0))
        actions.pack(fill="x")
        for text, handler in (
            ("推导计划", self.derive_plan),
            ("执行迁移", self.run_migration),
            ("刷新视图", self.refresh_view),
            ("崩溃检查", self.inspect_recovery),
            ("回退恢复", lambda: self.recover("rollback")),
            ("重建恢复", lambda: self.recover("rebuild")),
            ("重放校验", self.run_replay),
        ):
            ttk.Button(actions, text=text, command=handler).pack(
                side="left", padx=3, pady=4)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=6)
        self.tier_tab = ttk.Frame(notebook)
        self.plan_tab = ttk.Frame(notebook)
        self.log_tab = ttk.Frame(notebook)
        notebook.add(self.tier_tab, text="层状态与对象")
        notebook.add(self.plan_tab, text="迁移计划")
        notebook.add(self.log_tab, text="结构化日志")

        self.tier_text = self._text_area(self.tier_tab)
        self.plan_text = self._text_area(self.plan_tab)
        self.log_text = self._text_area(self.log_tab)
        self.status_var = self.tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief="sunken", anchor="w").pack(fill="x", side="bottom")

    def _text_area(self, parent):
        import tkinter.scrolledtext as st
        widget = st.ScrolledText(parent, wrap="word", font=("Consolas", 10))
        widget.pack(fill="both", expand=True)
        return widget

    def _set(self, widget, content: dict | str) -> None:
        text = content if isinstance(content, str) else json.dumps(
            content, ensure_ascii=False, indent=2, sort_keys=True)
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _post_log(self, entry: dict) -> None:
        self.events.put(entry)

    def _poll_events(self) -> None:
        try:
            while True:
                entry = self.events.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", json.dumps(
                    entry, ensure_ascii=False, sort_keys=True) + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(300, self._poll_events)

    def _run_async(self, name: str, job) -> None:
        self.status_var.set(f"{name} 执行中...")

        def worker():
            try:
                result = job()
                self.events.put({"event": "gui_job_done", "job": name,
                                 "ok": True, "result": result})
            except TierMoverError as exc:
                self.events.put({"event": "gui_job_done", "job": name,
                                 "ok": False, "error": exc.to_dict()})
            except Exception as exc:  # pragma: no cover - GUI 兜底
                self.events.put({"event": "gui_job_done", "job": name,
                                 "ok": False,
                                 "error": {"code": "E_UNKNOWN",
                                           "message": str(exc)}})

        threading.Thread(target=worker, daemon=True).start()

    def init_workspace(self) -> None:
        ws = Path(self.ws_var.get())

        def job():
            engine, ledger, manifest = bootstrap_workspace(ws)
            self.workspace, self.engine, self.ledger = ws, engine, ledger
            self.manifest = manifest
            self.logger = engine.logger
            return ledger.current_view()

        self._run_async("初始化工作区", job)
        self._after_job_refresh()

    def load_workspace(self) -> None:
        ws = Path(self.ws_var.get())
        paths = workspace_paths(ws)

        def job():
            logger = StructuredLogger(paths["logs"])
            manifest = load_manifest(paths["tiers"], paths["objects"],
                                     paths["seed"])
            engine = TierEngine(ws, logger=logger)
            ledger = Ledger(paths["ledger"])
            self.workspace, self.engine, self.ledger = ws, engine, ledger
            self.manifest, self.logger = manifest, logger
            return ledger.current_view()

        self._run_async("加载/校验", job)
        self._after_job_refresh()

    def _after_job_refresh(self) -> None:
        def check():
            try:
                entry = self.events.queue[-1]
            except IndexError:
                self.root.after(200, check)
                return
            if entry.get("event") == "gui_job_done":
                if entry["ok"]:
                    self._set(self.tier_text, entry["result"])
                    self.status_var.set(f"{entry['job']} 完成")
                else:
                    self._set(self.tier_text, entry["error"])
                    self.status_var.set(f"{entry['job']} 失败: "
                                        f"{entry['error'].get('code')}")
        self.root.after(200, check)

    def _require(self):
        if not (self.engine and self.ledger and self.manifest):
            raise TierMoverError("请先初始化或加载工作区", field="workspace")
        return self.engine, self.ledger, self.manifest

    def derive_plan(self) -> None:
        engine, ledger, manifest = self._require()

        def job():
            plan = derive_plan(manifest, ledger.placement())
            return json.loads(plan.to_json())

        self._run_async("推导计划", job)
        self._bind_result_to(self.plan_tab, self.plan_text)

    def run_migration(self) -> None:
        engine, ledger, manifest = self._require()

        def job():
            plan = engine.plan(ledger, manifest)
            conclusion = engine.execute(ledger, manifest, plan=plan)
            engine.reconcile_bytes(manifest, ledger)
            return conclusion

        self._run_async("执行迁移", job)
        self._bind_result_to(self.tier_tab, self.tier_text)

    def refresh_view(self) -> None:
        _engine, ledger, _manifest = self._require()
        self._set(self.tier_text, ledger.current_view())

    def inspect_recovery(self) -> None:
        engine, ledger, manifest = self._require()
        service = RecoveryService(engine, ledger, manifest, engine.logger)

        def job():
            return service.inspect().to_dict()

        self._run_async("崩溃检查", job)
        self._bind_result_to(self.tier_tab, self.tier_text)

    def recover(self, mode: str) -> None:
        engine, ledger, manifest = self._require()
        service = RecoveryService(engine, ledger, manifest, engine.logger)
        self._run_async(f"恢复({mode})",
                        lambda: service.recover(mode=mode).to_dict())
        self._bind_result_to(self.tier_tab, self.tier_text)

    def run_replay(self) -> None:
        _engine, ledger, manifest = self._require()
        service = ReplayService(ledger, manifest)
        self._run_async("重放校验", lambda: service.run().to_dict())
        self._bind_result_to(self.plan_tab, self.plan_text)

    def _bind_result_to(self, tab, widget) -> None:
        def check():
            if not self.events.queue:
                self.root.after(200, check)
                return
            entry = self.events.queue[-1]
            if entry.get("event") != "gui_job_done":
                self.root.after(200, check)
                return
            if entry["ok"]:
                self._set(widget, entry["result"])
                self.status_var.set(f"{entry['job']} 完成")
            else:
                self._set(widget, entry["error"])
                self.status_var.set(f"{entry['job']} 失败")
        self.root.after(200, check)


def main() -> int:
    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit(
            "当前 Python 环境缺少 tkinter，请在 Windows 官方 Python 安装中勾选 "
            "tcl/tk，或改用命令行 `python -m tier_mover.cli`。") from exc
    root = tk.Tk()
    TierMoverApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
