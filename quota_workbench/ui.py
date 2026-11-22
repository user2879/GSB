"""Tkinter diagnostics workbench. Tk imports are intentionally lazy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .snapshot import restore_snapshot, save_snapshot
from .session import WorkbenchSession


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def launch_ui(session: WorkbenchSession) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise WorkbenchError(
            "E_IO",
            "Tkinter is unavailable in this Python installation; the core CLI remains usable",
            object_type="ui",
            field="tkinter",
            details={"cause": str(exc)},
        ) from exc

    class DiagnosticsApp:
        def __init__(self, root: tk.Tk, current: WorkbenchSession):
            self.root = root
            self.session = current
            self.root.title("云原生多租户配额抢占诊断工作台")
            self.root.geometry("1440x880")
            self.root.minsize(1100, 680)
            self._configure_style()
            self._build_header()
            self._build_toolbar()
            self._build_tabs()
            self._build_status()
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.refresh()

        def _configure_style(self) -> None:
            style = ttk.Style(self.root)
            for theme in ("vista", "clam", "default"):
                if theme in style.theme_names():
                    style.theme_use(theme)
                    break
            style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
            style.configure("Subtitle.TLabel", font=("Microsoft YaHei UI", 9))
            style.configure("Status.TLabel", font=("Consolas", 9))
            style.configure("Treeview", font=("Consolas", 9), rowheight=24)
            style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

        def _build_header(self) -> None:
            frame = ttk.Frame(self.root, padding=(12, 10, 12, 4))
            frame.pack(fill="x")
            ttk.Label(frame, text="云原生多租户配额抢占诊断工作台", style="Title.TLabel").pack(
                anchor="w"
            )
            self.header_label = ttk.Label(frame, text="", style="Subtitle.TLabel")
            self.header_label.pack(anchor="w", pady=(4, 0))

        def _build_toolbar(self) -> None:
            frame = ttk.Frame(self.root, padding=(12, 4, 12, 6))
            frame.pack(fill="x")
            buttons = (
                ("运行下一事件", self.run_next),
                ("运行全部", self.run_all),
                ("刷新", self.refresh),
                ("重放校验", self.verify_replay),
                ("保存快照", self.save_snapshot),
                ("恢复快照", self.restore_snapshot),
            )
            for text, command in buttons:
                ttk.Button(frame, text=text, command=command).pack(side="left", padx=(0, 8))

        def _build_tabs(self) -> None:
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 4))

            self.tenant_tree = self._make_tree_tab(
                "租户分摊",
                (
                    ("tenant_id", "租户", 135),
                    ("priority", "优先级", 70),
                    ("share_bp", "份额万分比", 95),
                    ("share_floor_units", "份额下限", 80),
                    ("share_rounding_bonus_units", "余数补位", 85),
                    ("share_entitlement_units", "份额应得", 85),
                    ("reserve_units", "保留配额", 85),
                    ("reserve_uplift_units", "保留抬升", 85),
                    ("entitlement_units", "最终应得", 85),
                    ("workload_units", "工作负载占用", 105),
                    ("pressure_units", "压力占用", 85),
                    ("actual_units", "实际占用", 85),
                    ("excess_units", "超额量", 75),
                    ("deviation_units", "分摊偏差", 85),
                    ("share_quantization_error_scaled_1e4", "量化误差×1e4", 115),
                ),
            )
            self.preemption_tree = self._make_tree_tab(
                "本轮抢占",
                (
                    ("sequence", "序号", 55),
                    ("iteration", "轮次", 55),
                    ("tenant_id", "租户", 130),
                    ("workload_id", "工作负载", 150),
                    ("reclaimed_units", "回收单位", 85),
                    ("tenant_usage_before_units", "租户抢占前", 95),
                    ("tenant_usage_after_units", "租户抢占后", 95),
                    ("cluster_usage_before_units", "集群抢占前", 95),
                    ("cluster_usage_after_units", "集群抢占后", 95),
                    ("selection", "选择依据", 470),
                ),
            )
            self.iteration_tree = self._make_tree_tab(
                "收敛轨迹",
                (
                    ("iteration", "轮次", 60),
                    ("terminal_reason", "终止原因", 175),
                    ("matrix_digest", "矩阵摘要", 340),
                    ("cluster_excess", "集群超额", 90),
                    ("tenant_excess", "租户超额合计", 105),
                    ("remaining_gap", "剩余缺口", 85),
                    ("affected", "涉及租户", 260),
                ),
            )
            self.workload_tree = self._make_tree_tab(
                "工作负载状态",
                (
                    ("workload_id", "工作负载", 170),
                    ("tenant_id", "租户", 135),
                    ("requested_replicas", "请求副本", 85),
                    ("running_replicas", "运行副本", 85),
                    ("units_per_replica", "每副本单位", 95),
                    ("preemptible", "可抢占", 75),
                    ("startup_order", "启动序号", 85),
                    ("occupancy_units", "实际占用", 90),
                ),
            )
            self.ledger_tree = self._make_tree_tab(
                "账本事件",
                (
                    ("event_id", "事件", 55),
                    ("time_ms", "时刻ms", 75),
                    ("trigger_type", "触发类型", 145),
                    ("target", "目标", 175),
                    ("pressure_units", "压力值", 75),
                    ("terminal_code", "结论码", 120),
                    ("terminal_reason", "终止原因", 175),
                    ("iteration_count", "轮次", 55),
                    ("preemption_count", "抢占数", 60),
                    ("reclaimed_units", "回收", 60),
                    ("record_digest", "记录摘要", 270),
                ),
            )

            summary_frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(summary_frame, text="声明与恢复")
            self.summary_text = tk.Text(
                summary_frame,
                wrap="none",
                font=("Consolas", 9),
                background="#101722",
                foreground="#e6edf3",
                insertbackground="#e6edf3",
                relief="flat",
            )
            summary_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
            self.summary_text.configure(yscrollcommand=summary_scroll.set)
            self.summary_text.pack(side="left", fill="both", expand=True)
            summary_scroll.pack(side="right", fill="y")

        def _make_tree_tab(self, title: str, columns: tuple[tuple[str, str, int], ...]) -> Any:
            frame = ttk.Frame(self.notebook, padding=6)
            self.notebook.add(frame, text=title)
            keys = [item[0] for item in columns]
            tree = ttk.Treeview(frame, columns=keys, show="headings")
            for key, heading, width in columns:
                tree.heading(key, text=heading, anchor="w")
                tree.column(key, width=width, minwidth=45, anchor="w", stretch=True)
            y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            return tree

        def _build_status(self) -> None:
            frame = ttk.Frame(self.root, padding=(12, 2, 12, 8))
            frame.pack(fill="x")
            self.status_label = ttk.Label(frame, text="", style="Status.TLabel")
            self.status_label.pack(anchor="w")

        def _clear(self, tree: Any) -> None:
            children = tree.get_children()
            if children:
                tree.delete(*children)

        def _show_error(self, exc: Exception) -> None:
            if isinstance(exc, WorkbenchError):
                messagebox.showerror(exc.code, _pretty(exc.to_dict()), parent=self.root)
            else:
                messagebox.showerror("运行失败", str(exc), parent=self.root)

        def refresh(self) -> None:
            versions = self.session.declaration_versions()
            recovery = self.session.recovery_report()
            self.header_label.configure(
                text=(
                    f"集群单位 {self.session.declaration.cluster_total_units}    "
                    f"游标 {self.session.cursor}/{len(self.session.events)}    "
                    f"租户版本 {versions['tenant_set_version'][:12]}…    "
                    f"口径版本 {versions['quota_context_version'][:12]}…    "
                    f"恢复状态 {recovery['status']}"
                )
            )
            self._refresh_tenants()
            self._refresh_preemptions()
            self._refresh_iterations()
            self._refresh_workloads()
            self._refresh_ledger()
            self._refresh_summary()
            self.status_label.configure(
                text=(
                    f"账本头 {self.session.ledger.head_digest}    "
                    f"会话状态 {self.session.state_digest()[:24]}…"
                )
            )

        def _refresh_tenants(self) -> None:
            self._clear(self.tenant_tree)
            matrix = self.session.current_matrix()
            for row in matrix["tenants"]:
                self.tenant_tree.insert(
                    "",
                    "end",
                    values=[row[key] for key in self.tenant_tree["columns"]],
                )

        def _refresh_preemptions(self) -> None:
            self._clear(self.preemption_tree)
            for row in self.session.current_preemptions():
                basis = json.loads(row["selection_basis_json"])
                values = {
                    **row,
                    "selection": (
                        f"priority={basis['tenant_priority']}; preemptible={basis['preemptible']}; "
                        f"startup_order={basis['startup_order']}; guard={basis['reserve_guard']}"
                    ),
                }
                self.preemption_tree.insert(
                    "",
                    "end",
                    values=[values.get(key, "") for key in self.preemption_tree["columns"]],
                )

        def _refresh_iterations(self) -> None:
            self._clear(self.iteration_tree)
            for row in self.session.current_iterations():
                matrix = json.loads(row["matrix_json"])
                gap = json.loads(row["gap_json"]) if row["gap_json"] else None
                values = {
                    "iteration": row["iteration"],
                    "terminal_reason": row["terminal_reason"] or "",
                    "matrix_digest": row["matrix_digest"],
                    "cluster_excess": matrix["cluster_excess_units"],
                    "tenant_excess": matrix["tenant_excess_total_units"],
                    "remaining_gap": gap["remaining_gap_units"] if gap else 0,
                    "affected": ",".join(gap["affected_tenant_ids"]) if gap else "",
                }
                self.iteration_tree.insert(
                    "",
                    "end",
                    values=[values.get(key, "") for key in self.iteration_tree["columns"]],
                )

        def _refresh_workloads(self) -> None:
            self._clear(self.workload_tree)
            running = self.session.state.running_replicas
            for workload in self.session.declaration.sorted_workloads():
                replicas = running[workload.workload_id]
                values = {
                    "workload_id": workload.workload_id,
                    "tenant_id": workload.tenant_id,
                    "requested_replicas": workload.requested_replicas,
                    "running_replicas": replicas,
                    "units_per_replica": workload.units_per_replica,
                    "preemptible": "true" if workload.preemptible else "false",
                    "startup_order": workload.startup_order,
                    "occupancy_units": replicas * workload.units_per_replica,
                }
                self.workload_tree.insert(
                    "",
                    "end",
                    values=[values.get(key, "") for key in self.workload_tree["columns"]],
                )

        def _refresh_ledger(self) -> None:
            self._clear(self.ledger_tree)
            for row in self.session.ledger.event_rows():
                values = dict(row)
                values["target"] = row.get("target_tenant") or row.get("target_workload") or ""
                self.ledger_tree.insert(
                    "",
                    "end",
                    values=[values.get(key, "") for key in self.ledger_tree["columns"]],
                )

        def _refresh_summary(self) -> None:
            payload = {
                "declaration_versions": self.session.declaration_versions(),
                "declaration_path": self.session.spec_path,
                "events_path": self.session.events_path,
                "db_path": str(self.session.ledger.path),
                "recovery": self.session.recovery_report(),
                "ledger_head_digest": self.session.ledger.head_digest,
                "ledger_cursor": self.session.cursor,
                "event_count": len(self.session.events),
                "runtime_state": self.session.state.to_dict(self.session.declaration),
                "final_matrix": self.session.current_matrix(),
                "ledger_events": self.session.ledger.event_rows(),
            }
            self.summary_text.configure(state="normal")
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", _pretty(payload))
            self.summary_text.configure(state="disabled")

        def run_next(self) -> None:
            try:
                result = self.session.run_next()
                if result is None:
                    messagebox.showinfo("事件流结束", "没有剩余触发事件。", parent=self.root)
                elif result.unsatisfied:
                    messagebox.showwarning(
                        "E_UNSATISFIED",
                        _pretty(result.record["gap"]),
                        parent=self.root,
                    )
                self.refresh()
            except Exception as exc:
                self._show_error(exc)

        def run_all(self) -> None:
            try:
                while not self.session.completed:
                    result = self.session.run_next()
                    if result is None or result.unsatisfied:
                        if result is not None and result.unsatisfied:
                            messagebox.showwarning(
                                "E_UNSATISFIED",
                                _pretty(result.record["gap"]),
                                parent=self.root,
                            )
                        break
                self.refresh()
            except Exception as exc:
                self._show_error(exc)

        def verify_replay(self) -> None:
            try:
                report = self.session.verify_replay()
                messagebox.showinfo("重放校验", _pretty(report), parent=self.root)
            except Exception as exc:
                self._show_error(exc)

        def save_snapshot(self) -> None:
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title="保存只读会话快照",
                defaultextension=".json",
                filetypes=(("JSON snapshot", "*.json"), ("All files", "*.*")),
                initialdir=str(Path("demo_output")),
                initialfile="session.snapshot.json",
            )
            if not path:
                return
            try:
                snapshot = save_snapshot(self.session, path)
                messagebox.showinfo(
                    "快照已保存",
                    _pretty(
                        {
                            "path": str(snapshot.path),
                            "cursor": snapshot.ledger_cursor,
                            "head_digest": snapshot.ledger_head_digest,
                            "content_digest": snapshot.content_digest,
                            "readonly": True,
                        }
                    ),
                    parent=self.root,
                )
                self.refresh()
            except Exception as exc:
                self._show_error(exc)

        def restore_snapshot(self) -> None:
            snapshot_path = filedialog.askopenfilename(
                parent=self.root,
                title="选择只读快照",
                filetypes=(("JSON snapshot", "*.json"), ("All files", "*.*")),
            )
            if not snapshot_path:
                return
            db_path = filedialog.asksaveasfilename(
                parent=self.root,
                title="选择全新的恢复数据库路径",
                defaultextension=".sqlite3",
                filetypes=(("SQLite", "*.sqlite3"), ("All files", "*.*")),
                initialdir=str(Path("demo_output")),
                initialfile="restored.sqlite3",
            )
            if not db_path:
                return
            try:
                restored = restore_snapshot(
                    snapshot_path,
                    self._events_path,
                    db_path,
                    current_declaration_path=self._spec_path,
                )
                old = self.session
                self.session = restored.session
                self.spec_path = self.session.spec_path
                self.events_path = self.session.events_path or ""
                old.close()
                messagebox.showinfo(
                    "快照已整体替换",
                    _pretty(restored.change_report),
                    parent=self.root,
                )
                self.refresh()
            except Exception as exc:
                self._show_error(exc)

        @property
        def _spec_path(self) -> str | None:
            return getattr(self, "spec_path", None)

        @property
        def _events_path(self) -> str:
            return getattr(self, "events_path", "")

        def _on_close(self) -> None:
            try:
                self.session.close()
            finally:
                self.root.destroy()

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise WorkbenchError(
            "E_IO",
            "Tkinter cannot create a window in the current display/session",
            object_type="ui",
            field="display",
            details={"cause": str(exc)},
        ) from exc
    app = DiagnosticsApp(root, session)
    app.spec_path = session.spec_path
    app.events_path = session.events_path or ""
    root.mainloop()
