"""Tkinter diagnostics workbench.

The UI is deliberately a local, polling-free desktop view.  It never opens a
network socket and renders every table in a stable lexical order.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from .clock import LogicalClock
from .errors import SreError
from .loader import load_chain, load_trace
from .runner import BatchRunner
from .storage import SessionStore


class WorkbenchApp:
    def __init__(
        self,
        root: tk.Tk,
        chain_path: str | Path,
        trace_path: str | Path,
        db_path: str | Path,
        session_id: str,
    ) -> None:
        self.root = root
        self.chain_path = Path(chain_path)
        self.trace_path = Path(trace_path)
        self.db_path = Path(db_path)
        self.session_id = session_id
        self.chain = load_chain(self.chain_path)
        self.trace = load_trace(self.trace_path, self.chain)
        self.store = SessionStore(self.db_path)
        try:
            self.store.get_session(session_id)
        except SreError:
            self.store.create_session(self.chain, self.trace, session_id=session_id)
        self.clock = LogicalClock(0)
        self.runner = BatchRunner(
            self.store,
            self.session_id,
            self.chain,
            self.trace,
            clock=self.clock,
            follow_trace_time=True,
        )
        self.batch_size = tk.IntVar(value=4)
        self._build()
        self.refresh()

    def _build(self) -> None:
        self.root.title("服务调用链错误预算与熔断联动诊断工作台")
        self.root.geometry("1280x820")

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header,
            text=(
                f"服务链: {self.chain.version}  |  digest: {self.chain.digest[:16]}…  |  "
                f"会话: {self.session_id}"
            ),
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side="left")

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="批大小").pack(side="left")
        ttk.Spinbox(
            controls, from_=1, to=100, textvariable=self.batch_size, width=5
        ).pack(side="left", padx=(4, 10))
        ttk.Button(controls, text="推进一批", command=self.advance_batch).pack(side="left")
        ttk.Button(controls, text="保存快照", command=self.save_snapshot).pack(
            side="left", padx=6
        )
        ttk.Button(controls, text="整状态恢复", command=self.restore_snapshot).pack(
            side="left"
        )
        ttk.Button(controls, text="刷新", command=self.refresh).pack(side="left", padx=6)
        ttk.Label(controls, text="（依赖边根据链声明稳定排序）").pack(side="right")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        state_frame = ttk.Frame(notebook, padding=4)
        edge_frame = ttk.Frame(notebook, padding=4)
        fast_frame = ttk.Frame(notebook, padding=4)
        snapshot_frame = ttk.Frame(notebook, padding=4)
        notebook.add(state_frame, text="服务预算与熔断")
        notebook.add(edge_frame, text="依赖边")
        notebook.add(fast_frame, text="快速失败传递链")
        notebook.add(snapshot_frame, text="快照")

        state_columns = (
            "service",
            "timeout_ms",
            "window_hours",
            "ratio_bp",
            "min_samples",
            "window_samples",
            "failures",
            "budget_total",
            "budget_remaining",
            "consecutive_failures",
            "circuit_state",
            "cooldown_remaining_ms",
            "open_count",
        )
        self.state_tree = self._tree(state_frame, state_columns)

        self.edge_tree = self._tree(
            edge_frame,
            ("service", "upstream", "edge_direction"),
        )

        self.fast_tree = self._tree(
            fast_frame,
            (
                "call_index",
                "trigger_source",
                "trigger_reason",
                "service",
                "propagation_depth",
            ),
        )

        self.snapshot_tree = self._tree(
            snapshot_frame,
            (
                "snapshot_id",
                "cursor",
                "chain_version",
                "content_digest",
                "created_at",
            ),
        )

        self.status = tk.StringVar(value="就绪")
        ttk.Label(outer, textvariable=self.status, relief="sunken", anchor="w").pack(
            fill="x", pady=(8, 0)
        )

    def _tree(self, parent: ttk.Frame, columns: tuple[str, ...]) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=120, anchor="w", stretch=True)
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    @staticmethod
    def _clear(tree: ttk.Treeview) -> None:
        children = tree.get_children()
        if children:
            tree.delete(*children)

    def _display_clock_ms(self) -> int:
        return max(self.clock.now_ms(), self.runner.engine.last_logical_clock_ms)

    def refresh(self) -> None:
        self._clear(self.state_tree)
        for row in self.runner.engine.service_rows(self._display_clock_ms()):
            self.state_tree.insert(
                "",
                "end",
                values=tuple(row[column] for column in self.state_tree["columns"]),
            )

        self._clear(self.edge_tree)
        for service in sorted(self.chain.services, key=lambda item: item.id):
            for upstream in sorted(service.upstream):
                self.edge_tree.insert(
                    "",
                    "end",
                    values=(service.id, upstream, f"{upstream} → {service.id}"),
                )

        self._clear(self.fast_tree)
        conclusions = self.store.list_conclusions(
            self.session_id, valid_only=True, limit=20, descending=True
        )
        for row in sorted(conclusions, key=lambda item: item["call_index"]):
            decision = row["decision"]
            for item in decision.get("fast_fail_chain", []):
                self.fast_tree.insert(
                    "",
                    "end",
                    values=(
                        decision["call_index"],
                        item["trigger_source"],
                        item["trigger_reason"],
                        item["service"],
                        item["propagation_depth"],
                    ),
                )

        self._clear(self.snapshot_tree)
        for snapshot in self.store.list_snapshots(self.session_id):
            self.snapshot_tree.insert(
                "",
                "end",
                iid=snapshot["snapshot_id"],
                values=(
                    snapshot["snapshot_id"],
                    snapshot["cursor"],
                    snapshot["chain_version"],
                    snapshot["content_digest"][:20] + "…",
                    snapshot["created_at"],
                ),
            )

        session = self.store.get_session(self.session_id)
        self.status.set(
            "游标 {}/{}，已提交批次状态稳定；快照 {} 个；未知批次 {} 个".format(
                session["committed_cursor"],
                session["total_calls"],
                len(self.store.list_snapshots(self.session_id)),
                len(self.store.list_unknown(self.session_id)),
            )
        )

    def advance_batch(self) -> None:
        try:
            size = max(1, int(self.batch_size.get()))
            outcome = self.runner.run(max_batches=1, batch_size=size)
            if outcome.processed_calls:
                self.clock.follow(self.runner.engine.last_logical_clock_ms)
            self.refresh()
        except Exception as exc:
            messagebox.showerror("推进失败", str(exc))

    def save_snapshot(self) -> None:
        try:
            snapshot = self.runner.create_snapshot()
            self.refresh()
            messagebox.showinfo("快照已保存", snapshot.snapshot_id)
        except Exception as exc:
            messagebox.showerror("快照失败", str(exc))

    def restore_snapshot(self) -> None:
        selected = self.snapshot_tree.selection()
        if not selected:
            messagebox.showwarning("未选择快照", "请先选择一个快照")
            return
        try:
            result = self.store.restore_snapshot(
                selected[0], self.runner.engine, self.chain, self.trace
            )
            self.clock.set(max(0, self.runner.engine.last_logical_clock_ms))
            self.refresh()
            messagebox.showinfo(
                "整状态替换完成",
                "\n".join(
                    [
                        f"模式: {result.mode}",
                        f"原因: {result.reason}",
                        f"受影响服务: {', '.join(result.affected_services) or '无'}",
                        f"作废结论: {result.invalidated_conclusions}",
                    ]
                ),
            )
        except Exception as exc:
            messagebox.showerror("恢复失败", str(exc))


def launch_ui(
    chain_path: str | Path,
    trace_path: str | Path,
    db_path: str | Path,
    session_id: str,
) -> None:
    root = tk.Tk()
    app = WorkbenchApp(root, chain_path, trace_path, db_path, session_id)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.store.close(), root.destroy()))
    root.mainloop()
