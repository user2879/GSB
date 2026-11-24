from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from .app import run_scheduler
from .hashing import canonical_text
from .viewmodel import compare_cache_file


class DiagnosticUI:
    def __init__(self, root: tk.Tk, spec_path: Path, workdir: Path):
        self.root = root
        self.spec_path = spec_path
        self.workdir = workdir
        self.cache_path = workdir / "ui-cache.json"
        self.root.title("GPU 池调度 / 账本诊断台")
        self.root.geometry("1180x760")

        self.status = tk.StringVar(value=f"spec={spec_path} workdir={workdir}")
        self.compare_status = tk.StringVar(value="尚未执行权威重算")
        self.compare_detail = tk.StringVar(value="")
        self._build_controls()
        self._build_tabs()
        self.refresh_cached_view()
        self.root.after(1000, self._poll)

    def _build_controls(self) -> None:
        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill=tk.X)
        ttk.Label(frame, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Button(frame, text="刷新缓存", command=self.refresh_cached_view).pack(side=tk.RIGHT, padx=4)
        ttk.Button(frame, text="权威重算并逐字节比较", command=self.start_authoritative_compare).pack(side=tk.RIGHT, padx=4)
        ttk.Button(frame, text="运行演示调度", command=self.start_demo_run).pack(side=tk.RIGHT, padx=4)

        compare_frame = ttk.Frame(self.root, padding=(8, 0))
        compare_frame.pack(fill=tk.X)
        self.compare_label = ttk.Label(compare_frame, textvariable=self.compare_status, font=("Sans", 11, "bold"))
        self.compare_label.pack(anchor=tk.W)
        ttk.Label(compare_frame, textvariable=self.compare_detail, foreground="#8a1f11").pack(anchor=tk.W)

    def _build_tabs(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        pool_tab = ttk.Frame(notebook)
        shard_tab = ttk.Frame(notebook)
        preempt_tab = ttk.Frame(notebook)
        raw_tab = ttk.Frame(notebook)
        notebook.add(pool_tab, text="池配额水位")
        notebook.add(shard_tab, text="运行/排队/未知")
        notebook.add(preempt_tab, text="抢占与挂起")
        notebook.add(raw_tab, text="比对结果/原始缓存")

        columns = ("pool", "gpu_used", "gpu_cap", "gpu_remain", "ratio", "bound", "mem_used", "mem_cap", "tasks")
        self.pool_tree = ttk.Treeview(pool_tab, columns=columns, show="headings")
        headings = {
            "pool": "池/全局",
            "gpu_used": "GPU已用",
            "gpu_cap": "GPU容量",
            "gpu_remain": "GPU余量",
            "ratio": "定点比例",
            "bound": "误差上界",
            "mem_used": "显存已用MiB",
            "mem_cap": "显存容量MiB",
            "tasks": "活动任务",
        }
        for key, text in headings.items():
            self.pool_tree.heading(key, text=text)
            self.pool_tree.column(key, width=120, anchor=tk.W)
        self.pool_tree.pack(fill=tk.BOTH, expand=True)

        self.shard_text = tk.Text(shard_tab, wrap=tk.NONE, height=20)
        self.shard_text.pack(fill=tk.BOTH, expand=True)

        preempt_columns = ("round", "high", "victim", "set", "rule", "reason")
        self.preempt_tree = ttk.Treeview(preempt_tab, columns=preempt_columns, show="headings")
        for key, text in {
            "round": "轮次",
            "high": "高优先级",
            "victim": "被抢占",
            "set": "稳定被抢占集合",
            "rule": "选择规则",
            "reason": "原因",
        }.items():
            self.preempt_tree.heading(key, text=text)
            self.preempt_tree.column(key, width=170, anchor=tk.W)
        self.preempt_tree.pack(fill=tk.BOTH, expand=True)

        self.raw_text = tk.Text(raw_tab, wrap=tk.NONE)
        self.raw_text.pack(fill=tk.BOTH, expand=True)

    def _poll(self) -> None:
        self.refresh_cached_view()
        self.root.after(1000, self._poll)

    def _load_cache(self) -> dict[str, Any] | None:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.compare_status.set("缓存解析失败")
            self.compare_detail.set(str(exc))
            return None

    def refresh_cached_view(self) -> None:
        view = self._load_cache()
        for item in self.pool_tree.get_children():
            self.pool_tree.delete(item)
        for item in self.preempt_tree.get_children():
            self.preempt_tree.delete(item)
        if view is None:
            self.status.set(f"未发现 {self.cache_path}；可点击“运行演示调度”")
            return
        self.status.set(f"轮次={view.get('round_index')}  workdir={self.workdir}")
        watermarks = view.get("watermarks", {})
        global_gpu = watermarks.get("global", {}).get("gpu_quota", {})
        global_mem = watermarks.get("global", {}).get("memory_mib", {})
        self.pool_tree.insert(
            "",
            tk.END,
            values=(
                "__global__",
                global_gpu.get("used"),
                global_gpu.get("capacity"),
                global_gpu.get("remaining"),
                f"{global_gpu.get('ratio_fixed')}/{global_gpu.get('scale_factor')}",
                global_gpu.get("quant_error_bound", {}).get("absolute_ratio_upper_bound_exclusive"),
                global_mem.get("used"),
                global_mem.get("capacity"),
                watermarks.get("global", {}).get("active_tasks_sum"),
            ),
        )
        for pool in watermarks.get("pools", []):
            gpu = pool.get("gpu_physical", {})
            mem = pool.get("memory_mib", {})
            tasks = pool.get("active_tasks", {})
            self.pool_tree.insert(
                "",
                tk.END,
                values=(
                    pool.get("id"),
                    gpu.get("used"),
                    gpu.get("capacity"),
                    gpu.get("remaining"),
                    f"{gpu.get('ratio_fixed')}/{gpu.get('scale_factor')}",
                    gpu.get("quant_error_bound", {}).get("absolute_ratio_upper_bound_exclusive"),
                    mem.get("used"),
                    mem.get("capacity"),
                    f"{tasks.get('used')}/{tasks.get('capacity')}",
                ),
            )

        self.shard_text.delete("1.0", tk.END)
        lines = [
            f"状态计数: {canonical_text(view.get('counts', {}))}",
            "",
            f"运行中: {', '.join(view.get('running', [])) or '(none)'}",
            f"排队中: {', '.join(view.get('queued', [])) or '(none)'}",
            f"未知/强杀未提交: {', '.join(view.get('unknown', [])) or '(none)'}",
            f"被抢占任务: {', '.join(view.get('preempted_tasks', [])) or '(none)'}",
            "",
            "挂起状态:",
        ]
        for suspension in view.get("suspensions", []):
            lines.append(canonical_text(suspension))
        self.shard_text.insert(tk.END, "\n".join(lines))

        for preemption in view.get("preemptions", []):
            victim_set = preemption.get("victim_set_json")
            if victim_set is None:
                victim_set = preemption.get("victim_set", [])
            self.preempt_tree.insert(
                "",
                tk.END,
                values=(
                    preemption.get("round_index"),
                    preemption.get("high_task_id"),
                    preemption.get("victim_task_id"),
                    victim_set,
                    preemption.get("selection_rule"),
                    preemption.get("reason"),
                ),
            )

    def start_authoritative_compare(self) -> None:
        self.compare_status.set("权威线程重算中…")
        self.compare_label.configure(foreground="#7a5c00")

        def worker() -> None:
            try:
                from .spec import load_spec

                spec = load_spec(self.spec_path)
                result = compare_cache_file(spec, self.workdir, self.cache_path)
                self.root.after(0, lambda: self.render_compare(result))
            except Exception as exc:
                self.root.after(0, lambda: self.render_compare_error(exc))

        threading.Thread(target=worker, name="authoritative-recompute", daemon=True).start()

    def render_compare(self, result: dict[str, Any]) -> None:
        if result.get("byte_equal"):
            self.compare_status.set("✓ 缓存与权威重算的规范字节完全一致")
            self.compare_label.configure(foreground="#0a7a28")
            self.compare_detail.set(result["authoritative_sha256"])
        else:
            self.compare_status.set("✗ 检测到缓存/权威视图逐字节不一致，已标红")
            self.compare_label.configure(foreground="#c01818")
            diff = result.get("first_difference") or {}
            self.compare_detail.set(f"首个差异: {canonical_text(diff)}")
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.insert(tk.END, canonical_text(result, )[:200000])

    def render_compare_error(self, exc: Exception) -> None:
        self.compare_status.set("✗ 权威重算失败")
        self.compare_label.configure(foreground="#c01818")
        self.compare_detail.set(f"{type(exc).__name__}: {exc}")

    def start_demo_run(self) -> None:
        self.compare_status.set("后台线程启动真实回环调度…")

        def worker() -> None:
            try:
                report = run_scheduler(
                    self.spec_path,
                    self.workdir,
                    max_wall_seconds=45,
                    install_signals=False,
                )
                self.root.after(0, lambda: self.status.set("演示完成: " + canonical_text(report.get("counts", {}))))
                self.root.after(0, self.refresh_cached_view)
            except Exception as exc:
                self.root.after(0, lambda: self.render_compare_error(exc))

        threading.Thread(target=worker, name="ui-demo-run", daemon=True).start()


def launch_ui(spec_path: Path, workdir: Path) -> None:
    root = tk.Tk()
    DiagnosticUI(root, Path(spec_path), Path(workdir))
    root.mainloop()
