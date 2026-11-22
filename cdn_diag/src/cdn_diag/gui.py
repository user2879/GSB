"""CDN 边缘缓存一致性诊断台（tkinter 交互界面）。

左侧为场景控制与异常注入，中部为账本视图（条目/租约/事件），
底部为诊断报告与结构化日志。所有耗时操作在后台线程执行，
界面通过队列回显，不阻塞主循环。
"""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .demo import (  # noqa: E402
    EVOLVE_TICK_MS,
    clean_workspace,
    scenario_cold_start,
    scenario_evolution,
    scenario_snapshot_replay,
    workspace_root,
)
from .errors import DiagError  # noqa: E402
from .inject import cycle_derivation_override  # noqa: E402
from .logx import JsonlLogger  # noqa: E402
from .orchestrator import Orchestrator  # noqa: E402
from .report import build_report  # noqa: E402
from .spec import load_policy, load_requests  # noqa: E402
from . import inject  # noqa: E402


class DiagApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CDN 边缘缓存一致性诊断台")
        self.geometry("1180x760")
        self.ws = workspace_root()
        self.orch = None
        self.batches = None
        self._queue = queue.Queue()
        self._build_ui()
        self.after(100, self._drain_queue)

    # ---- UI 构建 ----
    def _build_ui(self):
        left = ttk.Frame(self, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        ttk.Label(left, text="场景控制", font=("", 12, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="启动系统（清理账本）", command=self.on_start).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="场景一：冷启动填充", command=self.on_cold).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="场景二：版本演进失效", command=self.on_evolve).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="场景三：快照恢复重放", command=self.on_snapshot).pack(fill=tk.X, pady=2)
        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="异常注入", font=("", 12, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="派生成环 (E_GRAPH)", command=lambda: self.on_inject("cycle")).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="快照截断 (E_SNAPSHOT)", command=lambda: self.on_inject("truncate")).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="摘要篡改 (E_VERSION)", command=lambda: self.on_inject("digest")).pack(fill=tk.X, pady=2)
        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="刷新账本视图", command=self.refresh_views).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="生成诊断报告", command=self.on_report).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="停止系统", command=self.on_stop).pack(fill=tk.X, pady=2)

        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self.tree_entries = self._make_tree(
            "缓存条目",
            ("node", "key", "etag", "ver", "ttl", "src"),
            ("节点", "缓存键", "ETag", "版本", "TTLms", "来源"),
        )
        self.tree_leases = self._make_tree(
            "租约",
            ("id", "node", "paths", "lver", "status"),
            ("租约ID", "节点", "路径集", "版本", "状态"),
        )
        self.tree_events = self._make_tree(
            "事件",
            ("seq", "kind", "payload"),
            ("序号", "类型", "负载"),
        )
        self.log = scrolledtext.ScrolledText(right, height=12, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=False, pady=4)

    def _make_tree(self, title, keys, headers):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=keys, show="headings")
        for k, h in zip(keys, headers):
            tree.heading(k, text=h)
            tree.column(k, width=120 if k != "payload" else 560)
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    # ---- 后台执行 ----
    def _run_bg(self, fn):
        def wrapper():
            try:
                fn()
            except DiagError as exc:
                self._queue.put(("log", f"[{exc.code}] {exc.message} {exc.details}"))
            except Exception as exc:  # noqa: BLE001
                self._queue.put(("log", f"[ERROR] {type(exc).__name__}: {exc}"))
            self._queue.put(("refresh", None))

        threading.Thread(target=wrapper, daemon=True).start()

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "refresh":
                    self.refresh_views()
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _append_log(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    # ---- 场景动作 ----
    def on_start(self):
        def work():
            clean_workspace(self.ws)
            logger = JsonlLogger(os.path.join(self.ws, "logs", "gui.jsonl"))
            policy = load_policy(os.path.join(self.ws, "samples", "policy.json"))
            self.batches = load_requests(os.path.join(self.ws, "samples", "requests.json"))
            self.orch = Orchestrator(self.ws, policy, logger)
            self.orch.start()
            self._queue.put(("log", f"系统已启动：内容源 + {len(self.orch.nodes)} 节点"))

        self._run_bg(work)

    def on_cold(self):
        self._run_bg(lambda: scenario_cold_start(self.orch, self.batches))

    def on_evolve(self):
        self._run_bg(lambda: scenario_evolution(self.orch))

    def on_snapshot(self):
        self._run_bg(lambda: scenario_snapshot_replay(self.orch))

    def on_inject(self, kind):
        def work():
            if kind == "cycle":
                self.orch.evolve_content(
                    "/index.html", "v2-index", b"INDEX-V2",
                    tick_ms=EVOLVE_TICK_MS,
                    derived_from=cycle_derivation_override(),
                )
            elif kind == "truncate":
                self.orch.save_snapshot()
                inject.truncate_snapshot(self.orch.snapshot_path)
                self.orch.restore()
            elif kind == "digest":
                self.orch.save_snapshot()
                inject.tamper_snapshot_digest(self.orch.snapshot_path)
                self.orch.restore()

        self._run_bg(work)

    def on_report(self):
        def work():
            text = build_report(self.orch.ledger)
            for line in text.splitlines():
                self._queue.put(("log", line))

        self._run_bg(work)

    def on_stop(self):
        def work():
            if self.orch:
                self.orch.stop()
                self.orch = None
            self._queue.put(("log", "系统已停止"))

        self._run_bg(work)

    # ---- 账本视图 ----
    def refresh_views(self):
        if not self.orch or not self.orch.ledger:
            return
        try:
            for tree in (self.tree_entries, self.tree_leases, self.tree_events):
                tree.delete(*tree.get_children())
            for node_id in sorted(self.orch.nodes):
                for e in self.orch.ledger.entries_for(node_id):
                    self.tree_entries.insert(
                        "", tk.END,
                        values=(node_id, e.cache_key, e.etag, e.entry_version,
                                e.ttl_ms, e.source),
                    )
            for lease in self.orch.ledger.leases():
                self.tree_leases.insert(
                    "", tk.END,
                    values=(lease["lease_id"], lease["node_id"], lease["paths"],
                            lease["lease_version"], lease["status"]),
                )
            for ev in self.orch.ledger.events():
                self.tree_events.insert(
                    "", tk.END, values=(ev["seq"], ev["kind"], str(ev["payload"]))
                )
        except Exception as exc:  # noqa: BLE001 - 账本可能正在替换
            self._append_log(f"[VIEW] {exc}")


def main():
    app = DiagApp()
    app.mainloop()


if __name__ == "__main__":
    main()
