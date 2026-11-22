from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from .errors import AppError
from .ledger import Ledger
from .planner import compile_plan
from .service import (
    execute_run,
    preview_ledger,
    recover_ledger,
    replay_restored_snapshot,
    save_ledger_snapshot,
    verify_run,
)
from .validation import validate_spec


def launch_gui(spec_path: str = "sample/spec.json", ledger_path: str = "artifacts/gui-ledger.sqlite3", snapshot_path: str = "artifacts/gui-snapshot.json") -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    class Workbench:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.spec_path = Path(spec_path)
            self.ledger_path = Path(ledger_path)
            self.snapshot_path = Path(snapshot_path)
            self.spec = None
            self.plan = None
            self.jobs: queue.Queue[tuple[Callable[[Any], None], Any, BaseException | None]] = queue.Queue()
            self.busy = False
            self.root.title("ConfigDist · 多集群配置分发工作台")
            self.root.geometry("1280x860")
            self._build()
            self._load_initial()
            self.root.after(100, self._poll_jobs)

        def _build(self) -> None:
            toolbar = ttk.Frame(self.root, padding=8)
            toolbar.pack(fill="x")
            self.status = tk.StringVar(value="就绪")
            ttk.Label(toolbar, textvariable=self.status).pack(side="left", padx=(0, 16))
            ttk.Button(toolbar, text="刷新账本预览", command=self.refresh_preview).pack(side="right", padx=4)

            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self._build_spec_tab(ttk.Frame(self.notebook))
            self._build_run_tab(ttk.Frame(self.notebook))
            self._build_ledger_tab(ttk.Frame(self.notebook))
            self._build_stale_tab(ttk.Frame(self.notebook))

        def _build_spec_tab(self, frame: ttk.Frame) -> None:
            self.notebook.add(frame, text="清单与计划")
            bar = ttk.Frame(frame, padding=6)
            bar.pack(fill="x")
            ttk.Button(bar, text="载入 JSON", command=self.choose_spec).pack(side="left", padx=3)
            ttk.Button(bar, text="保存 JSON", command=self.save_spec).pack(side="left", padx=3)
            ttk.Button(bar, text="校验并编译", command=self.compile_current).pack(side="left", padx=3)
            ttk.Button(bar, text="校验并开始运行", command=self.run_current).pack(side="left", padx=3)
            panes = ttk.Panedwindow(frame, orient="vertical")
            panes.pack(fill="both", expand=True, padx=6, pady=6)
            editor_frame = ttk.Frame(panes)
            self.editor = ScrolledText(editor_frame, wrap="none", font=("Consolas", 10), undo=True)
            self.editor.pack(fill="both", expand=True)
            panes.add(editor_frame, weight=3)
            plan_frame = ttk.Frame(panes)
            columns = ("cluster", "task", "shard", "revision", "units", "keys")
            self.plan_tree = ttk.Treeview(plan_frame, columns=columns, show="headings", height=10)
            for name, width in zip(columns, (130, 150, 70, 80, 70, 260)):
                self.plan_tree.heading(name, text=name)
                self.plan_tree.column(name, width=width, anchor="w")
            plan_scroll = ttk.Scrollbar(plan_frame, orient="vertical", command=self.plan_tree.yview)
            self.plan_tree.configure(yscrollcommand=plan_scroll.set)
            self.plan_tree.pack(side="left", fill="both", expand=True)
            plan_scroll.pack(side="right", fill="y")
            panes.add(plan_frame, weight=2)

        def _build_run_tab(self, frame: ttk.Frame) -> None:
            self.notebook.add(frame, text="调度运行")
            bar = ttk.Frame(frame, padding=6)
            bar.pack(fill="x")
            ttk.Button(bar, text="运行当前清单", command=self.run_current).pack(side="left", padx=3)
            ttk.Button(bar, text="权威重算", command=self.verify_current).pack(side="left", padx=3)
            ttk.Button(bar, text="保存快照", command=self.save_snapshot).pack(side="left", padx=3)
            ttk.Button(bar, text="恢复并重放", command=self.restore_and_replay).pack(side="left", padx=3)
            ttk.Button(bar, text="崩溃后恢复", command=self.recover_current).pack(side="left", padx=3)
            self.run_summary = ScrolledText(frame, height=9, font=("Consolas", 9))
            self.run_summary.pack(fill="x", padx=6, pady=4)
            columns = ("shard", "cluster", "task", "outcome", "revision", "wait", "elapsed", "digest")
            self.trace_tree = ttk.Treeview(frame, columns=columns, show="headings")
            for name, width in zip(columns, (260, 110, 130, 100, 80, 80, 80, 220)):
                self.trace_tree.heading(name, text=name)
                self.trace_tree.column(name, width=width, anchor="w")
            scroll = ttk.Scrollbar(frame, orient="vertical", command=self.trace_tree.yview)
            self.trace_tree.configure(yscrollcommand=scroll.set)
            self.trace_tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
            scroll.pack(side="right", fill="y", pady=6, padx=(0, 6))

        def _build_ledger_tab(self, frame: ttk.Frame) -> None:
            self.notebook.add(frame, text="当前账本")
            self.preview_summary = ScrolledText(frame, height=7, font=("Consolas", 9))
            self.preview_summary.pack(fill="x", padx=6, pady=4)
            columns = ("cluster", "task", "progress", "shard", "outcome", "rows", "byte_digest")
            self.ledger_tree = ttk.Treeview(frame, columns=columns, show="headings")
            for name, width in zip(columns, (120, 150, 100, 250, 100, 70, 250)):
                self.ledger_tree.heading(name, text=name)
                self.ledger_tree.column(name, width=width, anchor="w")
            scroll = ttk.Scrollbar(frame, orient="vertical", command=self.ledger_tree.yview)
            self.ledger_tree.configure(yscrollcommand=scroll.set)
            self.ledger_tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
            scroll.pack(side="right", fill="y", pady=6, padx=(0, 6))

        def _build_stale_tab(self, frame: ttk.Frame) -> None:
            self.notebook.add(frame, text="陈旧审计")
            ttk.Label(
                frame,
                text="以下分片属于旧 generation，仅供审计；不参与当前进度、摘要或预算汇总。",
                padding=8,
            ).pack(fill="x")
            columns = ("run", "shard", "outcome", "revision", "units")
            self.stale_tree = ttk.Treeview(frame, columns=columns, show="headings")
            for name, width in zip(columns, (250, 300, 100, 90, 80)):
                self.stale_tree.heading(name, text=name)
                self.stale_tree.column(name, width=width, anchor="w")
            scroll = ttk.Scrollbar(frame, orient="vertical", command=self.stale_tree.yview)
            self.stale_tree.configure(yscrollcommand=scroll.set)
            self.stale_tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
            scroll.pack(side="right", fill="y", pady=6, padx=(0, 6))

        def _load_initial(self) -> None:
            try:
                self.editor.delete("1.0", "end")
                self.editor.insert("1.0", self.spec_path.read_text(encoding="utf-8"))
                self.compile_current()
                self.refresh_preview()
            except Exception as exc:  # UI must remain usable after a bad starter file.
                self.show_error(exc)

        def choose_spec(self) -> None:
            selected = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
            if selected:
                self.spec_path = Path(selected)
                self._load_initial()

        def save_spec(self) -> None:
            selected = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if selected:
                Path(selected).write_text(self.editor.get("1.0", "end").rstrip() + "\n", encoding="utf-8")
                self.status.set(f"已保存 {selected}")

        def _parse_and_compile(self) -> tuple[Any, dict[str, Any]]:
            try:
                raw = json.loads(self.editor.get("1.0", "end"))
            except json.JSONDecodeError as exc:
                raise AppError(
                    "E_SPEC",
                    f"JSON 解析失败: {exc.msg}",
                    "json",
                    f"line {exc.lineno}, column {exc.colno}",
                ) from None
            spec = validate_spec(raw)
            return spec, compile_plan(spec)

        def compile_current(self) -> None:
            try:
                spec, plan = self._parse_and_compile()
                self.spec = spec
                self.plan = plan
                self._render_plan(plan)
                self.status.set(f"校验通过 · plan {plan['plan_digest'][:16]} · {len(plan['shards'])} 分片")
            except Exception as exc:
                self.show_error(exc)

        def _render_plan(self, plan: dict[str, Any]) -> None:
            self.plan_tree.delete(*self.plan_tree.get_children())
            for shard in plan["shards"]:
                self.plan_tree.insert(
                    "",
                    "end",
                    values=(
                        shard["cluster_id"],
                        shard["task_id"],
                        f"{shard['shard_index']:04d}",
                        shard["revision_version"],
                        shard["resource_units"],
                        ",".join(shard["config_keys"]),
                    ),
                )

        def run_current(self) -> None:
            if self.busy:
                return
            try:
                spec, plan = self._parse_and_compile()
            except Exception as exc:
                self.show_error(exc)
                return
            self.spec = spec
            self.plan = plan
            self._render_plan(plan)
            self.status.set("运行中：启动回环节点并执行三重预算调度…")

            def work() -> dict[str, Any]:
                return execute_run(spec, plan, self.ledger_path, snapshot_path=self.snapshot_path)

            self._background(work, self._run_done)

        def _run_done(self, result: dict[str, Any]) -> None:
            self.run_summary.delete("1.0", "end")
            summary = {
                "run_id": result["run_id"],
                "counts": result["counts"],
                "peak": result["peak"],
                "limits": result["limits"],
                "replacement": result["replacement"],
                "stale_shards_after": result["stale_shards_after"],
                "ledger": result["ledger"],
            }
            self.run_summary.insert("1.0", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            self.trace_tree.delete(*self.trace_tree.get_children())
            for trace in result["trace"]:
                digest = trace["aggregate"]["byte_digest"] if trace.get("aggregate") else ""
                wait = (trace["wait_ended_ms"] - trace["wait_started_ms"]) if trace["wait_ended_ms"] is not None else ""
                self.trace_tree.insert(
                    "",
                    "end",
                    values=(
                        trace["shard_id"],
                        trace["cluster_id"],
                        trace["task_id"],
                        trace["outcome"],
                        trace["revision_version"],
                        wait,
                        trace["elapsed_ms"] if trace["elapsed_ms"] is not None else "",
                        digest,
                    ),
                )
            self.status.set(f"运行完成 · {result['counts']}")
            self.refresh_preview()

        def verify_current(self) -> None:
            if self.busy:
                return
            self.status.set("权威重算：逐分片重放并比对缓存…")
            self._background(lambda: verify_run(self.ledger_path, use_nodes=True), self._verify_done)

        def _verify_done(self, result: dict[str, Any]) -> None:
            self.run_summary.delete("1.0", "end")
            self.run_summary.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            self.status.set(f"权威重算通过 · {result['records_checked']} 分片")

        def save_snapshot(self) -> None:
            if self.busy:
                return
            try:
                spec, _ = self._parse_and_compile()
            except Exception as exc:
                self.show_error(exc)
                return
            self._background(
                lambda: save_ledger_snapshot(spec, self.ledger_path, self.snapshot_path),
                lambda result: self.status.set(f"快照已保存 · {result['content_digest'][:16]}"),
            )

        def restore_and_replay(self) -> None:
            if self.busy:
                return
            self.status.set("整体替换恢复快照并重新下发…")
            self._background(lambda: replay_restored_snapshot(self.snapshot_path), self._replay_done)

        def _replay_done(self, result: dict[str, Any]) -> None:
            self.run_summary.delete("1.0", "end")
            self.run_summary.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            self.status.set(
                f"恢复重放通过 · {result['records_replayed']} 分片 · {result['restore']['mode']}"
            )

        def recover_current(self) -> None:
            if self.busy:
                return
            self._background(lambda: recover_ledger(self.ledger_path), self._recover_done)

        def _recover_done(self, result: dict[str, Any]) -> None:
            self.run_summary.delete("1.0", "end")
            self.run_summary.insert("1.0", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            self.status.set(
                f"恢复完成 · 已提交 {len(result['committed_shards'])} · 未知待重跑 {len(result['unknown_shards'])}"
            )
            self.refresh_preview()

        def refresh_preview(self) -> None:
            try:
                preview = preview_ledger(self.ledger_path)
            except Exception as exc:
                self.show_error(exc)
                return
            self.preview_summary.delete("1.0", "end")
            summary = {
                "run_id": preview["run_id"],
                "state": preview["state"],
                "completed_cursor": preview["completed_cursor"],
                "stale_shards_excluded": preview["stale_shards"],
            }
            self.preview_summary.insert("1.0", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            self.ledger_tree.delete(*self.ledger_tree.get_children())
            for group in preview["groups"]:
                progress = group["progress"]
                label = f"{progress['completed']}/{progress['total']} unknown={progress['unknown']}"
                for shard in group["shards"]:
                    self.ledger_tree.insert(
                        "",
                        "end",
                        values=(
                            group["cluster_id"],
                            group["task_id"],
                            label,
                            shard["shard_id"],
                            shard["outcome"],
                            shard["scan_rows"] if shard["scan_rows"] is not None else "",
                            shard["byte_digest"] or "",
                        ),
                    )
            self._refresh_stale()

        def _refresh_stale(self) -> None:
            self.stale_tree.delete(*self.stale_tree.get_children())
            try:
                with Ledger(self.ledger_path) as ledger:
                    for row in ledger.stale_shards():
                        self.stale_tree.insert(
                            "",
                            "end",
                            values=(
                                row["run_id"],
                                row["shard_id"],
                                row["outcome"],
                                row["revision_version"],
                                row["resource_units"],
                            ),
                        )
            except AppError:
                return

        def _background(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
            self.busy = True

            def runner() -> None:
                try:
                    result = work()
                except BaseException as exc:
                    self.jobs.put((done, None, exc))
                else:
                    self.jobs.put((done, result, None))

            threading.Thread(target=runner, daemon=True).start()

        def _poll_jobs(self) -> None:
            try:
                while True:
                    done, result, exc = self.jobs.get_nowait()
                    self.busy = False
                    if exc is not None:
                        self.show_error(exc)
                    else:
                        done(result)
            except queue.Empty:
                pass
            self.root.after(100, self._poll_jobs)

        def show_error(self, exc: BaseException) -> None:
            if isinstance(exc, AppError):
                payload = exc.to_dict()
                self.status.set(f"{exc.code} · {exc.location or ''} · {exc.message}")
                messagebox.showerror(exc.code, json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                self.status.set(f"E_INTERNAL · {exc}")
                messagebox.showerror("E_INTERNAL", str(exc))

    root = tk.Tk()
    Workbench(root)
    root.mainloop()
