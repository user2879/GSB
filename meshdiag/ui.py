from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import load_workspace
from .engine import Engine
from .errors import MeshError
from .replay import replay_snapshot


class Workbench:
    def __init__(self, root, workspace_path: Path) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.workspace_path = workspace_path
        self.workspace = load_workspace(workspace_path)
        self.engine = Engine(self.workspace)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.busy = False
        self.progress_total = 0
        self.progress_done = 0
        self._build()
        self.refresh()
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        tk = self.tk
        ttk = self.ttk
        self.root.title(f"MeshDiag · {self.workspace.workspace_id}")
        self.root.geometry("1320x860")
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(
            toolbar,
            text=f"Workspace: {self.workspace_path} · 127.0.0.1 only",
        ).pack(side="left")
        self.run_button = ttk.Button(toolbar, text="运行当前修订", command=self.run_round)
        self.run_button.pack(side="right", padx=3)
        self.recover_button = ttk.Button(
            toolbar, text="恢复选中快照", command=self.recover_selected
        )
        self.recover_button.pack(side="right", padx=3)
        self.replay_button = ttk.Button(
            toolbar, text="重放选中快照", command=self.replay_selected
        )
        self.replay_button.pack(side="right", padx=3)
        ttk.Button(toolbar, text="刷新", command=self.refresh).pack(side="right", padx=3)

        status_frame = ttk.Frame(outer)
        status_frame.pack(fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(status_frame, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.status_var = tk.StringVar(value="READY")
        ttk.Label(status_frame, textvariable=self.status_var, width=56).pack(
            side="left", padx=(8, 0)
        )

        pane = ttk.Panedwindow(outer, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, padding=4)
        middle = ttk.Frame(pane, padding=4)
        right = ttk.Frame(pane, padding=4)
        pane.add(left, weight=1)
        pane.add(middle, weight=1)
        pane.add(right, weight=2)

        ttk.Label(left, text="服务与实例").pack(anchor="w")
        self.services_tree = ttk.Treeview(
            left, columns=("kind", "endpoint", "limit"), show="headings", height=20
        )
        for column, title, width in (
            ("kind", "对象", 120),
            ("endpoint", "回环端点", 160),
            ("limit", "并发配额", 90),
        ):
            self.services_tree.heading(column, text=title)
            self.services_tree.column(column, width=width, anchor="w")
        self.services_tree.pack(fill="both", expand=True, pady=(3, 8))

        ttk.Label(left, text="熔断实例状态").pack(anchor="w")
        self.circuit_tree = ttk.Treeview(
            left,
            columns=("instance", "state", "failures", "remaining"),
            show="headings",
            height=10,
        )
        for column, title, width in (
            ("instance", "实例", 120),
            ("state", "状态", 90),
            ("failures", "连续失败", 80),
            ("remaining", "冷却剩余ms", 100),
        ):
            self.circuit_tree.heading(column, text=title)
            self.circuit_tree.column(column, width=width, anchor="w")
        self.circuit_tree.pack(fill="both", expand=True)

        ttk.Label(middle, text="规则依赖边").pack(anchor="w")
        self.deps_tree = ttk.Treeview(
            middle, columns=("source", "kind", "target"), show="headings", height=20
        )
        for column, title, width in (
            ("source", "源", 130),
            ("kind", "关系", 100),
            ("target", "目标", 130),
        ):
            self.deps_tree.heading(column, text=title)
            self.deps_tree.column(column, width=width, anchor="w")
        self.deps_tree.pack(fill="both", expand=True, pady=(3, 8))

        ttk.Label(middle, text="请求结局分布").pack(anchor="w")
        self.outcome_tree = ttk.Treeview(
            middle, columns=("outcome", "count"), show="headings", height=8
        )
        self.outcome_tree.heading("outcome", text="结局")
        self.outcome_tree.heading("count", text="数量")
        self.outcome_tree.column("outcome", width=150, anchor="w")
        self.outcome_tree.column("count", width=60, anchor="e")
        self.outcome_tree.pack(fill="both", expand=True)

        ttk.Label(right, text="规则修订与格式来源").pack(anchor="w")
        self.rules_tree = ttk.Treeview(
            right,
            columns=("rule", "revision", "format", "prefix", "target", "budget"),
            show="headings",
            height=12,
        )
        for column, title, width in (
            ("rule", "规则", 130),
            ("revision", "修订", 55),
            ("format", "来源", 60),
            ("prefix", "前缀", 100),
            ("target", "目标服务", 100),
            ("budget", "超时/重试/熔断/冷却", 220),
        ):
            self.rules_tree.heading(column, text=title)
            self.rules_tree.column(column, width=width, anchor="w")
        self.rules_tree.pack(fill="both", expand=True, pady=(3, 8))

        ttk.Label(right, text="当前快照列表").pack(anchor="w")
        self.snapshots_tree = ttk.Treeview(
            right,
            columns=("id", "status", "revision", "completed", "in_progress", "file"),
            show="headings",
            height=8,
        )
        for column, title, width in (
            ("id", "快照", 140),
            ("status", "状态", 80),
            ("revision", "修订", 55),
            ("completed", "完成", 55),
            ("in_progress", "进行中", 60),
            ("file", "文件", 280),
        ):
            self.snapshots_tree.heading(column, text=title)
            self.snapshots_tree.column(column, width=width, anchor="w")
        self.snapshots_tree.pack(fill="both", expand=True, pady=(3, 8))

        ttk.Label(right, text="一致性结论").pack(anchor="w")
        self.consistency = tk.Text(right, height=8, wrap="word", state="disabled")
        self.consistency.pack(fill="both", expand=True)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for button in (self.run_button, self.recover_button, self.replay_button):
            button.configure(state=state)

    def _run_background(self, name: str, operation: Callable[[], Any]) -> None:
        if self.busy:
            return
        self._set_busy(True)
        self.status_var.set(name)

        def target() -> None:
            try:
                result = operation()
                self.events.put({"event": "operation_done", "result": result})
            except MeshError as exc:
                self.events.put({"event": "operation_error", "error": exc.as_dict()})
            except Exception as exc:
                self.events.put(
                    {
                        "event": "operation_error",
                        "error": {"code": "E_INTERNAL", "message": str(exc)},
                    }
                )

        threading.Thread(target=target, daemon=True, name="meshdiag-ui").start()

    def run_round(self) -> None:
        round_id = f"ui-r{time.strftime('%Y%m%d-%H%M%S')}"

        def operation():
            return self.engine.run_round(
                round_id=round_id,
                progress=lambda event: self.events.put(event),
            )

        self.progress_done = 0
        self.progress_total = 0
        self.progress.configure(value=0)
        self._run_background(f"RUN {round_id}", operation)

    def _selected_snapshot(self) -> str | None:
        selection = self.snapshots_tree.selection()
        if not selection:
            self.status_var.set("请先选择一个快照")
            return None
        values = self.snapshots_tree.item(selection[0], "values")
        return str(values[0]) if values else None

    def recover_selected(self) -> None:
        snapshot_id = self._selected_snapshot()
        if snapshot_id is None:
            return
        self._run_background(
            f"RECOVER {snapshot_id}",
            lambda: self.engine.recover_snapshot(snapshot_id),
        )

    def replay_selected(self) -> None:
        snapshot_id = self._selected_snapshot()
        if snapshot_id is None:
            return
        self._run_background(
            f"REPLAY {snapshot_id}",
            lambda: replay_snapshot(
                self.engine,
                snapshot_id,
                current_revision=self.engine.load_workspace_revision(),
            ),
        )

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event.get("event")
                if kind == "round_started":
                    self.progress_total = int(event.get("total_requests", 0))
                    self.progress_done = 0
                    self.progress.configure(maximum=max(1, self.progress_total), value=0)
                    self.status_var.set(
                        f"RUNNING revision={event.get('revision_version')} "
                        f"0/{self.progress_total}"
                    )
                elif kind == "request_completed":
                    self.progress_done = int(event.get("completed", self.progress_done))
                    self.progress.configure(value=self.progress_done)
                    self.status_var.set(
                        f"RUNNING {self.progress_done}/{self.progress_total or '?'} "
                        f"{event.get('request_id')} {event.get('outcome')}"
                    )
                elif kind == "attempt":
                    self.status_var.set(
                        f"ATTEMPT {event.get('request_id')} "
                        f"#{event.get('attempt_no')} {event.get('instance_id')} "
                        f"{event.get('outcome')}"
                    )
                elif kind == "operation_done":
                    result = event.get("result")
                    self._set_busy(False)
                    self.status_var.set("DONE")
                    self._show_operation_result(result)
                    self.refresh()
                elif kind == "operation_error":
                    self._set_busy(False)
                    error = event.get("error", {})
                    self.status_var.set(str(error.get("code", "ERROR")))
                    self._set_consistency(
                        f"{error.get('code')} {error.get('message')}\n"
                        + "\n".join(
                            f"{row.get('path')}: {row.get('message')} "
                            f"expected={row.get('expected')} actual={row.get('actual')} "
                            f"basis={row.get('basis')}"
                            for row in error.get("errors", [])
                        )
                    )
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _show_operation_result(self, result: Any) -> None:
        if hasattr(result, "as_dict"):
            payload = result.as_dict()
            self._set_consistency(
                "\n".join(f"{key}: {payload[key]}" for key in sorted(payload))
            )
        elif isinstance(result, dict):
            self._set_consistency(
                "\n".join(f"{key}: {result[key]}" for key in sorted(result))
            )
        else:
            self._set_consistency(str(result))

    def _set_consistency(self, text: str) -> None:
        self.consistency.configure(state="normal")
        self.consistency.delete("1.0", "end")
        self.consistency.insert("1.0", text)
        self.consistency.configure(state="disabled")

    @staticmethod
    def _clear(tree) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def refresh(self) -> None:
        active_services = self.workspace.services
        try:
            active_revision, active_services = self.engine.load_active_revision_and_services()
        except MeshError:
            active_revision = None
        self._clear(self.services_tree)
        for service in sorted(active_services, key=lambda item: item.service_id):
            self.services_tree.insert(
                "",
                "end",
                values=(f"service:{service.service_id}", "-", service.max_concurrency),
            )
            for instance in sorted(service.instances, key=lambda item: item.instance_id):
                self.services_tree.insert(
                    "",
                    "end",
                    values=(f"  instance:{instance.instance_id}", instance.endpoint(), ""),
                )
        self._clear(self.circuit_tree)
        for row in self.engine.breakers.snapshot():
            self.circuit_tree.insert(
                "",
                "end",
                values=(
                    row["instance_id"],
                    row["state"],
                    row["consecutive_failures"],
                    row["remaining_cooldown_ms"],
                ),
            )
        revision = active_revision
        revision_error = None
        try:
            workspace_revision = self.engine.load_workspace_revision()
        except MeshError as exc:
            workspace_revision = None
            revision_error = exc
        self._clear(self.deps_tree)
        self._clear(self.rules_tree)
        if revision is not None:
            for source, target, kind in revision.dependency_graph.edges:
                self.deps_tree.insert("", "end", values=(source, kind, target))
            for rule in revision.rules:
                self.rules_tree.insert(
                    "",
                    "end",
                    values=(
                        rule.rule_id,
                        rule.revision_version,
                        f"v{rule.format_version}",
                        rule.prefix,
                        rule.target_service,
                        f"{rule.timeout_ms}/{rule.retries}/"
                        f"{rule.circuit_threshold or '-'}/{rule.cooldown_ms}",
                    ),
                )
        self._clear(self.outcome_tree)
        view = self.engine.ledger.current_view()
        counts: dict[str, int] = {}
        if view is not None:
            for row in view["requests"]:
                outcome = str(row["outcome"])
                counts[outcome] = counts.get(outcome, 0) + 1
        for outcome in sorted(counts):
            self.outcome_tree.insert("", "end", values=(outcome, counts[outcome]))
        self._clear(self.snapshots_tree)
        snapshots = self.engine.snapshots.list()
        for row in snapshots:
            self.snapshots_tree.insert(
                "",
                "end",
                values=(
                    row["snapshot_id"],
                    row["status"],
                    row["revision_version"],
                    row["completed"],
                    row["in_progress"],
                    row["file"],
                ),
            )
        state = self.engine.ledger.current_state()
        if revision_error is not None:
            consistency = f"E_RULE 当前修订不可加载: {revision_error}"
        elif state is None:
            consistency = "NO_CURRENT_ROUND 尚无治理结论"
        elif (
            workspace_revision is not None
            and state.get("revision_version") == workspace_revision.revision_version
            and state.get("revision_digest") == workspace_revision.digest
        ):
            consistency = (
                "CONSISTENT 当前账本修订与工作区修订一致；"
                "历史轮次仅作审计且 stale=1 不进入当前视图"
            )
        elif revision is not None:
            consistency = (
                "SNAPSHOT_ACTIVE 当前账本/界面使用快照恢复的修订 "
                f"r{revision.revision_version}/{revision.digest[:12]}；"
                "工作区修订文件与活动账本不一致，运行新轮次前须递增版本"
            )
        else:
            consistency = (
                "CURRENT_MISMATCH 当前账本版本与工作区修订不一致，"
                "需要新版本轮次整体替换或在界面恢复快照"
            )
        self._set_consistency(consistency)

    def close(self) -> None:
        self.engine.close()


def run_ui(workspace_path: Path) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        print(f"E_TKINTER: {exc}", file=__import__("sys").stderr)
        return 3
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"E_TK_DISPLAY: {exc}", file=__import__("sys").stderr)
        return 3
    try:
        app = Workbench(root, workspace_path)
    except Exception as exc:
        messagebox.showerror("MeshDiag", str(exc))
        root.destroy()
        return 2

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0
