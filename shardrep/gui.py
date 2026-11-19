"""Tkinter diagnostic workstation for the local shard repair middleware."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .audit import StructuredLogger
from .ledger import Ledger
from .packing import PackStore, build_pack_tasks
from .propagation import propagate_invalidation
from .registry import ReplicaRegistry
from .repair import apply_and_report
from .scheduler import PackScheduler
from .spec import load_spec


def launch_gui(spec_path: str = "samples/spec.json") -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:  # pragma: no cover - Windows official Python includes Tk.
        raise RuntimeError("Tkinter is unavailable; install the Windows official Python build") from exc

    class DiagnosticApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("ShardRep · 分片副本修复诊断台")
            self.root.geometry("1180x760")
            self.spec = None
            self.registry = None
            self.affected: tuple[str, ...] = ()
            self.pack_store = None
            self.logger = None
            self.ledger = None

            self.spec_var = tk.StringVar(value=spec_path)
            self.slot_var = tk.IntVar(value=2)
            self.timeout_var = tk.IntVar(value=1500)
            self.shard_var = tk.StringVar(value="node-b:s4")
            self.kind_var = tk.StringVar(value="flip")
            self.status_var = tk.StringVar(value="准备就绪；所有路径均为本地工作区相对路径。")

            toolbar = ttk.Frame(root, padding=8)
            toolbar.pack(fill="x")
            ttk.Label(toolbar, text="规格 JSON").pack(side="left")
            ttk.Entry(toolbar, textvariable=self.spec_var, width=40).pack(side="left", padx=6)
            ttk.Button(toolbar, text="强校验", command=self.validate).pack(side="left", padx=3)
            ttk.Label(toolbar, text="全局槽位").pack(side="left", padx=(14, 3))
            ttk.Spinbox(toolbar, from_=1, to=16, textvariable=self.slot_var, width=5).pack(side="left")
            ttk.Label(toolbar, text="超时 ms").pack(side="left", padx=(10, 3))
            ttk.Spinbox(toolbar, from_=1, to=60000, textvariable=self.timeout_var, width=8).pack(side="left")

            action = ttk.Frame(root, padding=(8, 0, 8, 8))
            action.pack(fill="x")
            ttk.Label(action, text="物理分片").pack(side="left")
            ttk.Entry(action, textvariable=self.shard_var, width=26).pack(side="left", padx=5)
            ttk.Label(action, text="损坏类型").pack(side="left")
            ttk.Combobox(
                action,
                textvariable=self.kind_var,
                values=["truncate", "flip", "replace", "delete"],
                state="readonly",
                width=10,
            ).pack(side="left", padx=5)
            ttk.Button(action, text="注入并推导影响", command=self.inject_and_propagate).pack(side="left", padx=4)
            ttk.Button(action, text="增量重打包", command=self.repack).pack(side="left", padx=4)
            ttk.Button(action, text="修复与报告", command=self.repair).pack(side="left", padx=4)
            ttk.Button(action, text="运行完整演示", command=self.run_demo).pack(side="left", padx=12)
            ttk.Button(action, text="恢复快照", command=self.restore_snapshot).pack(side="left", padx=4)

            self.notebook = ttk.Notebook(root)
            self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.summary_text = self._text_tab("摘要")
            self.trace_tree = self._tree_tab(
                "失效传播", ("逻辑分片", "序号", "深度", "直接", "传递轨迹")
            )
            self.task_tree = self._tree_tab(
                "调度", ("任务", "节点", "结局", "开始 ms", "结束 ms", "排队 ms", "字节", "版本", "原因")
            )
            self.report_text = self._text_tab("修复报告")
            self.log_text = self._text_tab("事件日志")

            status = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w", padding=5)
            status.pack(fill="x", side="bottom")
            self.root.after(100, self.validate)

        def _text_tab(self, title: str) -> tk.Text:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            text = tk.Text(frame, wrap="none", font=("Consolas", 10))
            scroll_y = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
            text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
            text.grid(row=0, column=0, sticky="nsew")
            scroll_y.grid(row=0, column=1, sticky="ns")
            scroll_x.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            return text

        def _tree_tab(self, title: str, columns: tuple[str, ...]) -> ttk.Treeview:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            tree = ttk.Treeview(frame, columns=columns, show="headings")
            for column in columns:
                tree.heading(column, text=column)
                tree.column(column, width=130 if column != "传递轨迹" else 520, anchor="w")
            scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scroll.set)
            tree.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            return tree

        def _set_text(self, widget: tk.Text, value: Any) -> None:
            widget.delete("1.0", "end")
            widget.insert("1.0", json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))

        def _runtime(self) -> None:
            path = Path(self.spec_var.get()).resolve()
            root = path.parent.parent if path.parent.name == "samples" else Path.cwd()
            log_dir = root / "run" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.ledger = Ledger(root / "run" / "shardrep.sqlite3")
            self.logger = StructuredLogger(log_dir / "events.jsonl", self.ledger)
            self.spec = load_spec(path)
            self.registry = ReplicaRegistry.load(self.spec, logger=self.logger)
            self.pack_store = PackStore(root / "run" / "packs", self.spec)

        def validate(self) -> None:
            try:
                self._runtime()
                assert self.spec is not None and self.registry is not None
                value = {
                    "group": self.spec.group_id,
                    "group_version": self.spec.group_version,
                    "mirror_graph_version": self.spec.mirror_graph_version,
                    "nodes": [node.node_id for node in self.spec.nodes],
                    "topological_order": list(self.spec.topo_order),
                    "registry": self.registry.summary(),
                    "integrity_issues": self.registry.integrity_issues,
                }
                self._set_text(self.summary_text, value)
                self.status_var.set(
                    f"校验通过：{len(self.spec.logical_shards)} 个逻辑分片，"
                    f"{len(self.registry.records)} 个物理副本"
                )
            except Exception as exc:
                messagebox.showerror("校验失败", str(exc))
                self.status_var.set(f"校验失败：{exc}")

        def inject_and_propagate(self) -> None:
            self.validate()
            if self.spec is None or self.registry is None:
                return
            try:
                injection = self.registry.inject(
                    shard_id=self.shard_var.get().strip(), kind=self.kind_var.get(), sequence=0
                )
                propagation = propagate_invalidation(self.spec, [injection.logical_id])
                self.registry.mark_affected(propagation.affected)
                self.affected = propagation.affected
                for item in self.trace_tree.get_children():
                    self.trace_tree.delete(item)
                for entry in propagation.entries:
                    trace_text = " | ".join(
                        "→".join(path.path) for path in entry.cause
                    )
                    self.trace_tree.insert(
                        "",
                        "end",
                        values=(entry.logical_id, entry.seq, entry.depth, "是" if entry.direct else "否", trace_text),
                    )
                self._set_text(self.summary_text, propagation.as_dict())
                self.status_var.set(
                    f"已注入 {injection.shard_id}，影响 {len(propagation.affected)} 个逻辑分片；可执行增量重打包"
                )
            except Exception as exc:
                messagebox.showerror("注入/传播失败", str(exc))

        def repack(self) -> None:
            if self.spec is None or self.registry is None or self.pack_store is None:
                self.validate()
            if self.spec is None or self.registry is None or self.pack_store is None:
                return
            targets = self.affected or self.spec.topo_order
            quotas = {node.node_id: node.pack_quota for node in self.spec.nodes}
            tasks = build_pack_tasks(
                self.spec,
                targets,
                timeout_ms=self.timeout_var.get(),
                simulate_ms=20,
                max_retries=1,
                quotas=quotas,
            )
            try:
                result = PackScheduler(
                    store=self.pack_store,
                    global_slots=self.slot_var.get(),
                    node_quotas=quotas,
                    logger=self.logger,
                    ledger=self.ledger,
                    run_id="gui-run",
                ).run(tasks)
                for item in self.task_tree.get_children():
                    self.task_tree.delete(item)
                for entry in result.entries:
                    self.task_tree.insert(
                        "",
                        "end",
                        values=(
                            entry.task_id,
                            entry.node_id,
                            entry.outcome,
                            "" if entry.started_at_ms is None else round(entry.started_at_ms, 3),
                            "" if entry.ended_at_ms is None else round(entry.ended_at_ms, 3),
                            round(entry.queue_ms, 3),
                            entry.completed_bytes,
                            entry.version or "",
                            entry.skip_reason,
                        ),
                    )
                self._set_text(
                    self.summary_text,
                    {
                        "targets": list(targets),
                        "schedule": result.as_dict(),
                        "pack_manifest": self.pack_store.manifest(),
                    },
                )
                self.status_var.set(
                    f"重打包完成：全局峰值 {result.max_global_observed}/{result.global_slots}"
                )
            except Exception as exc:
                messagebox.showerror("调度失败", str(exc))

        def repair(self) -> None:
            if self.spec is None or self.registry is None:
                self.validate()
            if self.spec is None or self.registry is None:
                return
            try:
                report = apply_and_report(self.spec, self.registry, logger=self.logger)
                self.registry.clear_pending()
                self._set_text(self.report_text, report)
                self.status_var.set(
                    "修复报告：完全 {full} / 部分 {partial} / 不可修复 {unrecoverable}".format(**report["counts"])
                )
            except Exception as exc:
                messagebox.showerror("修复失败", str(exc))

        def run_demo(self) -> None:
            from .demo import run_demo

            self.status_var.set("正在运行确定性完整演示…")
            spec = self.spec_var.get()
            root = Path(spec).resolve().parent.parent

            def worker() -> None:
                try:
                    report = run_demo(root, simulate_ms=12, timeout_ms=1500, max_retries=1)
                    self.root.after(0, lambda: self._demo_done(report))
                except Exception as exc:
                    self.root.after(0, lambda: messagebox.showerror("演示失败", str(exc)))

            threading.Thread(target=worker, daemon=True).start()

        def _demo_done(self, report: dict[str, Any]) -> None:
            self._set_text(self.report_text, report)
            self._set_text(
                self.summary_text,
                {
                    "stable_digest": report["stable_digest"],
                    "stage2_affected": report["stage2_incremental"]["affected"],
                    "snapshot_restore": report["snapshot_restore"]["restore"],
                    "replay": report["replay"],
                },
            )
            self._refresh_log()
            self.validate()
            self.status_var.set("完整演示完成；日志与 SQLite 账本已落盘。")

        def restore_snapshot(self) -> None:
            from .snapshot import SnapshotManager

            self.validate()
            if self.spec is None or self.registry is None:
                return
            root = self.spec.source_path.parent.parent
            snapshot = root / "demo-run" / "snapshots" / "session.json"
            state = root / "demo-run" / "state" / "gui_restored_state.json"
            try:
                result = SnapshotManager(snapshot).restore(
                    spec=self.spec,
                    registry=self.registry,
                    state_path=state,
                    logger=self.logger,
                )
                self._set_text(self.summary_text, result)
                self.status_var.set(
                    "快照恢复完成：整体替换" if result["replaced"] else "快照与当前磁盘状态一致"
                )
            except Exception as exc:
                messagebox.showerror("快照恢复失败", str(exc))

        def _refresh_log(self) -> None:
            if self.logger is None or self.logger.jsonl_path is None:
                return
            self.log_text.delete("1.0", "end")
            if self.logger.jsonl_path.exists():
                self.log_text.insert("1.0", self.logger.jsonl_path.read_text(encoding="utf-8"))

    root = tk.Tk()
    app = DiagnosticApp(root)
    root.mainloop()

