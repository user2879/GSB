"""Tkinter diagnostic workbench with deterministic rendering order."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

from .engine import ExecutionResult, execute
from .gui_model import build_gui_model
from .ledger import Ledger
from .model import RuntimeConfig, Spec


def launch(spec: Spec, config: RuntimeConfig, path: str, ledger_path: str,
           snapshot_path: str, check: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable in this Python installation. On Windows use the "
            "official python.org build with the 'tcl/tk and IDLE' option."
        ) from exc

    root = tk.Tk()
    root.title("TxDiag - 事务隔离与锁等待诊断工作台")
    root.geometry("1280x820")
    if check:
        root.update_idletasks()
        root.destroy()
        return

    app = DiagnosticsWindow(root, spec, config, path, ledger_path, snapshot_path,
                            tk, ttk, messagebox)
    app.pack(fill="both", expand=True)
    root.mainloop()


class DiagnosticsWindow:
    def __init__(self, root: Any, spec: Spec, config: RuntimeConfig, path: str,
                 ledger_path: str, snapshot_path: str, tk: Any, ttk: Any,
                 messagebox: Any) -> None:
        self.root = root
        self.spec = spec
        self.config = config
        self.path = path
        self.ledger_path = ledger_path
        self.snapshot_path = snapshot_path
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.result: ExecutionResult | None = None
        self.model: dict[str, Any] | None = None
        self.message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.running = False

        self.frame = ttk.Frame(root)
        self._build_toolbar()
        self._build_tabs()
        self.root.after(150, self._poll_messages)
        self.root.after(50, self.run_now)

    def pack(self, **kwargs: Any) -> None:
        self.frame.pack(**kwargs)

    def _build_toolbar(self) -> None:
        bar = self.ttk.Frame(self.frame)
        bar.pack(fill="x", padx=8, pady=6)
        self.status = self.tk.StringVar(value="准备运行")
        self.ttk.Button(bar, text="运行 / 重放校验", command=self.run_now).pack(side="left")
        self.ttk.Label(bar, text="  路径:").pack(side="left")
        self.ttk.Label(bar, text=self.path).pack(side="left")
        self.ttk.Label(bar, text=f"  账本: {self.ledger_path}").pack(side="left")
        self.ttk.Label(bar, text=f"  快照: {self.snapshot_path}").pack(side="left")
        self.ttk.Label(bar, textvariable=self.status).pack(side="right")

    def _build_tabs(self) -> None:
        self.notebook = self.ttk.Notebook(self.frame)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        self.graph_tab = self.ttk.Frame(self.notebook)
        self.lock_tab = self.ttk.Frame(self.notebook)
        self.deadlock_tab = self.ttk.Frame(self.notebook)
        self.quota_tab = self.ttk.Frame(self.notebook)
        self.isolation_tab = self.ttk.Frame(self.notebook)
        self.events_tab = self.ttk.Frame(self.notebook)
        self.summary_tab = self.ttk.Frame(self.notebook)
        self.notebook.add(self.graph_tab, text="事务与行依赖图")
        self.notebook.add(self.lock_tab, text="锁持有 / 等待")
        self.notebook.add(self.deadlock_tab, text="等待图 / 死锁")
        self.notebook.add(self.quota_tab, text="配额占用时间线")
        self.notebook.add(self.isolation_tab, text="隔离级别结论")
        self.notebook.add(self.events_tab, text="事件轨迹")
        self.notebook.add(self.summary_tab, text="运行摘要")

        self.canvas = self.tk.Canvas(self.graph_tab, background="#fbfbfd",
                                     highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_graph())

        self.lock_tree = self._tree(self.lock_tab, ("row", "txid", "modes", "state"),
                                    ("行键", "事务", "模式", "状态/事件"))
        self.deadlock_text = self.tk.Text(self.deadlock_tab, wrap="none", font=("Consolas", 10))
        self.deadlock_text.pack(fill="both", expand=True)
        self.quota_tree = self._tree(self.quota_tab, (
            "round", "logical_ms", "running", "global", "isolation", "queued"),
            ("轮次", "逻辑毫秒", "运行集合", "全局", "隔离配额", "排队"))
        self.isolation_tree = self._tree(self.isolation_tab, (
            "txid", "isolation", "outcome", "reads", "fixed"),
            ("事务", "隔离级别", "结果", "读取结论", "固定版本"))
        self.event_tree = self._tree(self.events_tab, (
            "round", "seq", "at", "event", "txid", "row", "detail"),
            ("轮次", "序号", "逻辑毫秒", "事件", "事务", "行键", "详情"))
        self.summary_text = self.tk.Text(self.summary_tab, wrap="none",
                                         font=("Consolas", 10))
        self.summary_text.pack(fill="both", expand=True)

    def _tree(self, parent: Any, columns: tuple[str, ...], headings: tuple[str, ...]) -> Any:
        frame = self.ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings")
        for column, heading in zip(columns, headings):
            tree.heading(column, text=heading)
            tree.column(column, width=130, anchor="w")
        y_scroll = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def run_now(self) -> None:
        if self.running:
            return
        self.running = True
        self.status.set("运行中...")
        thread = threading.Thread(target=self._execute_background, name="txdiag-gui-run",
                                  daemon=True)
        thread.start()

    def _execute_background(self) -> None:
        ledger: Ledger | None = None
        try:
            ledger = Ledger(self.ledger_path, self.snapshot_path, self.spec,
                            self.config, self.path)
            opened = ledger.open()
            result = execute(self.spec, self.config, self.path, hooks=ledger)
            ledger.complete(result)
            session = ledger.session_info()
            self.message_queue.put({
                "kind": "result",
                "result": result,
                "opened": opened,
                "session": session,
            })
        except Exception as exc:
            self.message_queue.put({"kind": "error", "error": repr(exc)})
        finally:
            if ledger is not None:
                ledger.close()

    def _poll_messages(self) -> None:
        try:
            while True:
                message = self.message_queue.get_nowait()
                if message["kind"] == "result":
                    self.running = False
                    self.result = message["result"]
                    self.model = build_gui_model(self.spec, self.result)
                    self._render(message["opened"], message["session"])
                    self.status.set(f"{self.result.status} / {len(self.result.rounds)} 轮")
                else:
                    self.running = False
                    self.status.set("运行失败")
                    self.messagebox.showerror("TxDiag", message["error"])
        except queue.Empty:
            pass
        self.root.after(150, self._poll_messages)

    def _clear(self, tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _render(self, opened: dict[str, Any], session: dict[str, Any]) -> None:
        assert self.model is not None
        self._draw_graph()
        self._clear(self.lock_tree)
        for item in self.model["locks"]:
            self.lock_tree.insert("", "end", values=(
                item["row"], item["txid"], item["modes"], item["state"]))
        lines: list[str] = []
        for item in self.model["wait_graph"]["rounds"]:
            lines.append(f"round={item['round']} nodes={item['nodes']} edges={item['edges']} cycles={item['cycles']}")
        lines.append("")
        for item in self.model["deadlocks"]:
            lines.append(f"round={item['round']} cycle={item['cycle']} victim={item['victim']} basis={item['basis']} edges={item['edges']}")
        self.deadlock_text.delete("1.0", "end")
        self.deadlock_text.insert("1.0", "\n".join(lines) + "\n")
        self._clear(self.quota_tree)
        for item in self.model["quota_timeline"]:
            self.quota_tree.insert("", "end", values=(
                item["round"], item["logical_ms"], ",".join(item["running"]),
                f"{item['global_used']}/{item['global_limit']}",
                json.dumps(item["isolation"], ensure_ascii=False, sort_keys=True),
                ",".join(item["queued"])))
        self._clear(self.isolation_tree)
        for item in self.model["isolation"]:
            reads = ",".join(f"{read['row']}:v{read.get('version')}" for read in item["reads"])
            fixed = json.dumps(item["fixed_reads"], ensure_ascii=False, sort_keys=True)
            self.isolation_tree.insert("", "end", values=(
                item["txid"], item["isolation"], item["outcome"], reads, fixed))
        self._clear(self.event_tree)
        for item in self.model["events"]:
            self.event_tree.insert("", "end", values=(
                item["round"], item["event_seq"], item["at_ms"], item["event"],
                item["txid"] or "", item["row"] or "", item["detail"]))
        summary = {
            "status": self.result.status if self.result else None,
            "path": self.path,
            "opened": opened,
            "session": session,
            "result": self.model["summary"],
        }
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", json.dumps(summary, ensure_ascii=False,
                                                  indent=2, sort_keys=True) + "\n")

    def _draw_graph(self) -> None:
        if self.model is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 900)
        tx_x = 110
        row_x = 540
        top = 54
        tx_gap = max(66, min(96, (canvas.winfo_height() - top - 20)
                             // max(1, len(self.model["transactions"]))))
        row_gap = max(42, min(70, (canvas.winfo_height() - top - 20)
                              // max(1, len(self.model["row_nodes"]))))
        tx_positions: dict[str, tuple[int, int]] = {}
        row_positions: dict[str, tuple[int, int]] = {}
        canvas.create_text(tx_x, 20, text="事务（按开始序号）", font=("Segoe UI", 11, "bold"))
        canvas.create_text(row_x + 150, 20, text="行键与依赖边", font=("Segoe UI", 11, "bold"))
        for idx, tx in enumerate(self.model["transactions"]):
            x = tx_x
            y = top + idx * tx_gap
            tx_positions[tx["txid"]] = (x, y)
            canvas.create_rectangle(x - 86, y - 18, x + 90, y + 18, fill="#e8f0fe",
                                    outline="#5178a8")
            canvas.create_text(x - 80, y, anchor="w", font=("Consolas", 9),
                               text=f'{tx["txid"]}  s={tx["start_seq"]}')
            canvas.create_text(x + 84, y, anchor="e", font=("Consolas", 8),
                               text=tx["isolation"])
        for idx, row in enumerate(self.model["row_nodes"]):
            x = row_x + (idx % 3) * 175
            y = top + (idx // 3) * row_gap
            row_positions[row["row"]] = (x, y)
            canvas.create_oval(x - 62, y - 15, x + 70, y + 15, fill="#fff5df",
                               outline="#9d7a2e")
            canvas.create_text(x, y, font=("Consolas", 9), text=row["row"])
        for edge in self.model["dependency_edges"]:
            start = row_positions.get(edge["from"])
            end = row_positions.get(edge["to"])
            if start and end:
                canvas.create_line(*start, *end, arrow="last", fill="#9d7a2e", width=1)
        for edge in self.model["access_edges"]:
            start = tx_positions.get(edge["from"])
            end = row_positions.get(edge["to"])
            if start and end:
                canvas.create_line(start[0] + 90, start[1], end[0] - 64, end[1],
                                   fill="#c5cbe0", dash=(2, 3))
        for idx, cycle in enumerate(self.model["wait_graph"]["active_cycles"]):
            canvas.create_text(width - 150, 44 + idx * 20, fill="#a40000",
                               font=("Consolas", 9), text=f"当前等待环: {cycle}")
