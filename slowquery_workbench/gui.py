"""Tkinter desktop workbench for Windows/local desktop use."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from .errors import WorkbenchError
from .samples import DEFAULT_SAMPLES_DIR, load_queries
from .workbench import Workbench


class MultilineDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, title: str, prompt: str) -> None:
        super().__init__(parent)
        self.title(title)
        self.geometry("720x420")
        self.transient(parent)
        self.grab_set()
        self.result: str | None = None
        ttk.Label(self, text=prompt).pack(anchor="w", padx=10, pady=(10, 4))
        self.text = tk.Text(self, wrap="word", undo=True)
        self.text.pack(fill="both", expand=True, padx=10, pady=4)
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=10, pady=10)
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="执行", command=self._accept).pack(side="right", padx=6)
        self.wait_window(self)

    def _accept(self) -> None:
        self.result = self.text.get("1.0", "end").strip()
        self.destroy()


class WorkbenchApp:
    def __init__(self, initial_session: str | None = None, samples_dir: str = str(DEFAULT_SAMPLES_DIR)) -> None:
        self.root = tk.Tk()
        self.root.title("SQLite 慢查询诊断与改写工作台")
        self.root.geometry("1380x860")
        self.samples_dir = Path(samples_dir)
        self.wb: Workbench | None = None
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if initial_session:
            self._open_path(Path(initial_session))

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill="x")
        actions = [
            ("新建会话", self._new_session),
            ("打开", self._open_session),
            ("另存为", self._save_as),
            ("关闭", self._close_session),
            ("建表 SQL", self._create_table),
            ("载入样本", self._load_samples),
            ("保存快照", self._save_snapshot),
            ("恢复快照", self._restore_snapshot),
            ("重放校验", self._verify_replay),
        ]
        for text, command in actions:
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=3)
        self.session_label = ttk.Label(toolbar, text="未打开会话", padding=(12, 0))
        self.session_label.pack(side="left", padx=12)

        self.version_label = ttk.Label(self.root, text="data_version=- index_version=- stats_version=-", padding=(8, 2))
        self.version_label.pack(fill="x")

        pane = ttk.Panedwindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(pane)
        right = ttk.Frame(pane)
        pane.add(left, weight=3)
        pane.add(right, weight=2)

        query_frame = ttk.LabelFrame(left, text="SELECT 查询（单表或两表 INNER JOIN）")
        query_frame.pack(fill="x", padx=2, pady=2)
        self.query_text = tk.Text(query_frame, height=8, wrap="word", undo=True)
        self.query_text.pack(fill="x", padx=6, pady=6)
        self.query_text.insert(
            "1.0",
            "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id "
            "WHERE o.status = 'paid' AND o.amount >= 10 AND o.amount >= 100 "
            "AND c.city IN ('Shanghai', 'Beijing') ORDER BY o.id",
        )
        query_buttons = ttk.Frame(query_frame)
        query_buttons.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(query_buttons, text="载入示例查询", command=self._load_sample_query).pack(side="left")
        ttk.Button(query_buttons, text="执行并记录", command=self._execute_query).pack(side="left", padx=6)
        ttk.Button(query_buttons, text="生成候选并回放", command=self._diagnose).pack(side="left")

        controls = ttk.LabelFrame(left, text="阈值、超时与预算（毫秒 / 扫描行数）")
        controls.pack(fill="x", padx=2, pady=2)
        self.threshold_ms = tk.StringVar(value="1")
        self.threshold_rows = tk.StringVar(value="1000")
        self.timeout_ms = tk.StringVar(value="500")
        self.budget_ms = tk.StringVar(value="4000")
        for label, variable, width in (
            ("慢查询毫秒阈值", self.threshold_ms, 8),
            ("扫描行阈值", self.threshold_rows, 8),
            ("单次超时", self.timeout_ms, 8),
            ("总回放预算", self.budget_ms, 8),
        ):
            ttk.Label(controls, text=label).pack(side="left", padx=(8, 2), pady=6)
            ttk.Entry(controls, textvariable=variable, width=width).pack(side="left")
        ttk.Button(controls, text="保存阈值", command=self._save_thresholds).pack(side="left", padx=8)
        ttk.Label(controls, text="扫描行口径：FROM/JOIN/WHERE 实际进入管线的行数").pack(side="left", padx=8)

        exec_frame = ttk.LabelFrame(left, text="最近执行观测（诊断账本真值）")
        exec_frame.pack(fill="both", expand=True, padx=2, pady=2)
        self.execution_tree = ttk.Treeview(
            exec_frame,
            columns=("kind", "elapsed", "scanned", "rows", "status", "execution"),
            show="headings",
            height=8,
        )
        for column, title, width in (
            ("kind", "类型", 140),
            ("elapsed", "耗时ms", 90),
            ("scanned", "扫描行", 80),
            ("rows", "结果行", 80),
            ("status", "状态", 90),
            ("execution", "execution_id", 210),
        ):
            self.execution_tree.heading(column, text=title)
            self.execution_tree.column(column, width=width, anchor="w")
        self.execution_tree.pack(fill="both", expand=True, padx=6, pady=6)

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True, padx=2, pady=2)
        self._build_slow_tab(tabs)
        self._build_conclusion_tab(tabs)
        self._build_index_tab(tabs)
        self._build_event_tab(tabs)

        log_frame = ttk.LabelFrame(self.root, text="操作与错误输出")
        log_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_frame, height=7, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_slow_tab(self, tabs: ttk.Notebook) -> None:
        frame = ttk.Frame(tabs)
        tabs.add(frame, text="慢查询")
        self.slow_tree = ttk.Treeview(
            frame,
            columns=("query", "elapsed", "scanned", "rows", "thresholds"),
            show="headings",
        )
        for column, title, width in (
            ("query", "query_id", 280),
            ("elapsed", "耗时ms", 90),
            ("scanned", "扫描行", 80),
            ("rows", "结果行", 80),
            ("thresholds", "判定阈值", 130),
        ):
            self.slow_tree.heading(column, text=title)
            self.slow_tree.column(column, width=width, anchor="w")
        self.slow_tree.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(frame, text="重新按当前阈值分类", command=self._refresh_slow).pack(anchor="e", padx=4, pady=4)

    def _build_conclusion_tab(self, tabs: ttk.Notebook) -> None:
        frame = ttk.Frame(tabs)
        tabs.add(frame, text="诊断结论")
        self.conclusion_tree = ttk.Treeview(
            frame,
            columns=("query", "state", "best", "speedup", "versions", "thresholds"),
            show="headings",
            height=11,
        )
        for column, title, width in (
            ("query", "query_id", 260),
            ("state", "状态", 70),
            ("best", "最优候选", 150),
            ("speedup", "加速比", 70),
            ("versions", "版本 d/i/s", 100),
            ("thresholds", "阈值", 110),
        ):
            self.conclusion_tree.heading(column, text=title)
            self.conclusion_tree.column(column, width=width, anchor="w")
        self.conclusion_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.conclusion_tree.bind("<<TreeviewSelect>>", self._show_conclusion_detail)
        self.conclusion_detail = tk.Text(frame, height=10, wrap="word")
        self.conclusion_detail.pack(fill="both", expand=False, padx=4, pady=(0, 4))

    def _build_index_tab(self, tabs: ttk.Notebook) -> None:
        frame = ttk.Frame(tabs)
        tabs.add(frame, text="索引与统计")
        self.index_tree = ttk.Treeview(
            frame,
            columns=("name", "table", "columns", "unique"),
            show="headings",
        )
        for column, title, width in (
            ("name", "索引", 190),
            ("table", "表", 100),
            ("columns", "列", 180),
            ("unique", "唯一", 60),
        ):
            self.index_tree.heading(column, text=title)
            self.index_tree.column(column, width=width, anchor="w")
        self.index_tree.pack(fill="both", expand=True, padx=4, pady=4)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", padx=4, pady=4)
        ttk.Button(buttons, text="新增索引", command=self._add_index).pack(side="left")
        ttk.Button(buttons, text="删除选中索引", command=self._drop_index).pack(side="left", padx=4)
        ttk.Button(buttons, text="刷新统计", command=self._refresh_stats).pack(side="left")
        ttk.Button(buttons, text="追加订单测试行", command=self._append_order_row).pack(side="right")

    def _build_event_tab(self, tabs: ttk.Notebook) -> None:
        frame = ttk.Frame(tabs)
        tabs.add(frame, text="版本失效事件")
        self.event_tree = ttk.Treeview(
            frame,
            columns=("source", "old", "new", "tables", "queries", "reason"),
            show="headings",
        )
        for column, title, width in (
            ("source", "版本来源", 140),
            ("old", "旧", 55),
            ("new", "新", 55),
            ("tables", "表", 100),
            ("queries", "受影响查询数", 100),
            ("reason", "原因", 300),
        ):
            self.event_tree.heading(column, text=title)
            self.event_tree.column(column, width=width, anchor="w")
        self.event_tree.pack(fill="both", expand=True, padx=4, pady=4)

    def _log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _error(self, exc: WorkbenchError) -> None:
        details = json.dumps(exc.details, ensure_ascii=False, indent=2) if exc.details is not None else ""
        self._log(f"[{exc.code}] {exc.message}\n{details}")
        messagebox.showerror("业务错误", f"{exc.code}: {exc.message}\n\n{details}")

    def _set_workbench(self, workbench: Workbench | None) -> None:
        self.wb = workbench
        self._refresh_all()

    def _require(self) -> Workbench | None:
        if self.wb is None:
            messagebox.showinfo("未打开会话", "请先新建或打开诊断会话。")
            return None
        return self.wb

    def _new_session(self) -> None:
        path = filedialog.asksaveasfilename(
            title="新建诊断会话",
            defaultextension=".sqlite",
            filetypes=[("SQLite session", "*.sqlite;*.db"), ("All files", "*.*")],
        )
        if not path:
            return
        name = simpledialog.askstring("会话名称", "请输入可命名会话名称：", initialvalue=Path(path).stem, parent=self.root)
        if name is None:
            return
        try:
            if self.wb is not None:
                self.wb.close()
            self._set_workbench(Workbench.create(path, name=name, overwrite=True))
            self._log(f"已创建会话 {path}")
        except WorkbenchError as exc:
            self._error(exc)

    def _open_session(self) -> None:
        path = filedialog.askopenfilename(
            title="打开诊断会话",
            filetypes=[("SQLite session", "*.sqlite;*.db"), ("All files", "*.*")],
        )
        if path:
            self._open_path(Path(path))

    def _open_path(self, path: Path) -> None:
        try:
            if self.wb is not None:
                self.wb.close()
            self._set_workbench(Workbench.open(path))
            self._log(f"已打开会话 {path}")
        except WorkbenchError as exc:
            self._error(exc)

    def _save_as(self) -> None:
        wb = self._require()
        if wb is None:
            return
        path = filedialog.asksaveasfilename(
            title="会话另存为",
            defaultextension=".sqlite",
            filetypes=[("SQLite session", "*.sqlite;*.db")],
        )
        if not path:
            return
        try:
            new_wb = wb.save_as(path)
            wb.close()
            self._set_workbench(new_wb)
            self._log(f"会话已另存为 {path}")
        except WorkbenchError as exc:
            self._error(exc)

    def _close_session(self) -> None:
        if self.wb is None:
            return
        try:
            self.wb.close()
            self._log(f"已关闭会话 {self.wb.path}")
        finally:
            self.wb = None
            self._refresh_all()

    def _create_table(self) -> None:
        if self._require() is None:
            return
        dialog = MultilineDialog(self.root, "建表 SQL", "仅允许 CREATE TABLE 语句：")
        if not dialog.result:
            return
        try:
            self.wb.create_table(dialog.result)
            self._log("建表成功")
            self._refresh_all()
        except WorkbenchError as exc:
            self._error(exc)

    def _load_samples(self) -> None:
        if self._require() is None:
            return
        try:
            result = self.wb.initialize_samples(self.samples_dir)
            self._log("确定性样本已载入：" + json.dumps(result, ensure_ascii=False, default=str))
            self._refresh_all()
        except WorkbenchError as exc:
            self._error(exc)

    def _load_sample_query(self) -> None:
        try:
            entries = load_queries(self.samples_dir)
            if not entries:
                return
            self.query_text.delete("1.0", "end")
            self.query_text.insert("1.0", entries[0]["sql"])
        except WorkbenchError as exc:
            self._error(exc)

    def _query_text(self) -> str:
        return self.query_text.get("1.0", "end").strip()

    def _execute_query(self) -> None:
        wb = self._require()
        if wb is None:
            return
        try:
            execution = wb.execute_select(self._query_text(), timeout_ms=int(self.timeout_ms.get()))
            self._log(json.dumps(execution, ensure_ascii=False, indent=2, default=str))
            self._refresh_all()
        except (ValueError, WorkbenchError) as exc:
            if isinstance(exc, WorkbenchError):
                self._error(exc)
            else:
                messagebox.showerror("参数错误", "超时必须是整数毫秒")

    def _diagnose(self) -> None:
        wb = self._require()
        if wb is None:
            return
        try:
            result = wb.diagnose(
                self._query_text(),
                timeout_ms=int(self.timeout_ms.get()),
                budget_ms=int(self.budget_ms.get()),
                threshold_ms=int(self.threshold_ms.get()),
                threshold_rows=int(self.threshold_rows.get()),
            )
            self._log(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            self._refresh_all()
        except (ValueError, WorkbenchError) as exc:
            if isinstance(exc, WorkbenchError):
                self._error(exc)
            else:
                messagebox.showerror("参数错误", "阈值、超时和预算必须是整数")

    def _save_thresholds(self) -> None:
        wb = self._require()
        if wb is None:
            return
        try:
            result = wb.set_thresholds(int(self.threshold_ms.get()), int(self.threshold_rows.get()))
            self._log("阈值已保存，历史观测保持原始值：" + json.dumps(result, ensure_ascii=False))
            self._refresh_all()
        except (ValueError, WorkbenchError) as exc:
            if isinstance(exc, WorkbenchError):
                self._error(exc)
            else:
                messagebox.showerror("参数错误", "阈值必须是非负整数")

    def _refresh_all(self) -> None:
        for tree in (self.execution_tree, self.slow_tree, self.conclusion_tree, self.index_tree, self.event_tree):
            for item in tree.get_children():
                tree.delete(item)
        self.conclusion_detail.delete("1.0", "end")
        if self.wb is None:
            self.session_label.configure(text="未打开会话")
            self.version_label.configure(text="data_version=- index_version=- stats_version=-")
            return
        try:
            info = self.wb.info()
            versions = info["versions"]
            self.session_label.configure(text=f"{info['session_name']}  |  {info['path']}")
            self.version_label.configure(
                text=(
                    f"data_version={versions['data_version']}  index_version={versions['index_version']}  "
                    f"stats_version={versions['stats_version']}  |  {info['clock_basis']}"
                )
            )
            thresholds = info["thresholds"]
            self.threshold_ms.set(str(thresholds["threshold_ms"]))
            self.threshold_rows.set(str(thresholds["threshold_rows"]))
            for event in self.wb.execution_history()[-100:]:
                self.execution_tree.insert(
                    "",
                    "end",
                    values=(
                        event["kind"],
                        f"{event['elapsed_ms']:.3f}",
                        event["scanned_rows"],
                        event["result_rows"],
                        event["outcome"],
                        event["execution_id"],
                    ),
                )
            self._refresh_slow()
            for conclusion in self.wb.conclusions():
                self.conclusion_tree.insert(
                    "",
                    "end",
                    iid=conclusion["query_id"],
                    values=(
                        conclusion["query_id"],
                        "陈旧" if conclusion["is_stale"] else "有效",
                        conclusion.get("optimal_candidate_id") or "-",
                        f"{conclusion['speedup']:.2f}" if conclusion.get("speedup") is not None else "-",
                        f"{conclusion['data_version']}/{conclusion['index_version']}/{conclusion['stats_version']}",
                        f"{conclusion['threshold_ms']}ms/{conclusion['threshold_rows']}行",
                    ),
                )
            for index in self.wb.indexes():
                self.index_tree.insert(
                    "",
                    "end",
                    iid=index["name"],
                    values=(index["name"], index["table"], ", ".join(index["columns"]), index["unique"]),
                )
            for event in self.wb.change_events():
                self.event_tree.insert(
                    "",
                    "end",
                    values=(
                        event["source_version_field"],
                        event["old_value"],
                        event["new_value"],
                        ", ".join(event["affected_tables"]),
                        len(event["affected_query_ids"]),
                        event["reason"],
                    ),
                )
        except WorkbenchError as exc:
            self._error(exc)

    def _refresh_slow(self) -> None:
        for item in self.slow_tree.get_children():
            self.slow_tree.delete(item)
        if self.wb is None:
            return
        for item in self.wb.slow_queries():
            iid = f"{item['execution_id']}"
            if self.slow_tree.exists(iid):
                continue
            self.slow_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    item["query_id"],
                    f"{item['elapsed_ms']:.3f}",
                    item["scanned_rows"],
                    item["result_rows"],
                    f"{item['threshold_ms']}ms/{item['threshold_rows']}行",
                ),
            )

    def _show_conclusion_detail(self, _event: Any = None) -> None:
        if self.wb is None:
            return
        selected = self.conclusion_tree.selection()
        if not selected:
            return
        conclusion = self.wb.session.conclusion(selected[0])
        self.conclusion_detail.delete("1.0", "end")
        if conclusion:
            self.conclusion_detail.insert("1.0", json.dumps(conclusion, ensure_ascii=False, indent=2, default=str))

    def _add_index(self) -> None:
        wb = self._require()
        if wb is None:
            return
        table = simpledialog.askstring("新增索引", "表名：", parent=self.root)
        if not table:
            return
        columns = simpledialog.askstring("新增索引", "列名（逗号分隔）：", parent=self.root)
        if not columns:
            return
        name = simpledialog.askstring("新增索引", "索引名（可留空）：", parent=self.root) or None
        try:
            self.wb.add_index(table.strip(), [item.strip() for item in columns.split(",") if item.strip()], name=name)
            self._log("索引已新增，相关结论已按版本失效")
            self._refresh_all()
        except WorkbenchError as exc:
            self._error(exc)

    def _drop_index(self) -> None:
        if self._require() is None:
            return
        selected = self.index_tree.selection()
        if not selected:
            messagebox.showinfo("未选择", "请选择一个索引。")
            return
        try:
            self.wb.drop_index(selected[0])
            self._log(f"索引已删除：{selected[0]}")
            self._refresh_all()
        except WorkbenchError as exc:
            self._error(exc)

    def _refresh_stats(self) -> None:
        wb = self._require()
        if wb is None:
            return
        table = simpledialog.askstring("刷新统计", "表名（留空表示全部）：", parent=self.root)
        try:
            self.wb.refresh_statistics(table.strip() or None if table is not None else None)
            self._log("统计快照已刷新并推进 stats_version")
            self._refresh_all()
        except WorkbenchError as exc:
            self._error(exc)

    def _append_order_row(self) -> None:
        if self._require() is None:
            return
        try:
            self.wb.append_rows(
                "orders",
                ["id", "customer_id", "status", "amount", "created_at"],
                [(999999, 1, "paid", 123.45, "2024-12-31")],
            )
            self._log("已追加订单测试行并推进 data_version")
            self._refresh_all()
        except WorkbenchError as exc:
            self._error(exc)

    def _save_snapshot(self) -> None:
        wb = self._require()
        if wb is None:
            return
        path = filedialog.asksaveasfilename(
            title="保存会话快照",
            defaultextension=".sqlite",
            filetypes=[("SQLite snapshot", "*.sqlite;*.db")],
        )
        if path:
            try:
                snapshot = wb.save_snapshot(path)
                self._log(f"快照已保存：{snapshot}")
            except WorkbenchError as exc:
                self._error(exc)

    def _restore_snapshot(self) -> None:
        wb = self._require()
        if wb is None:
            return
        path = filedialog.askopenfilename(
            title="恢复会话快照",
            filetypes=[("SQLite snapshot", "*.sqlite;*.db")],
        )
        if not path:
            return
        try:
            result = wb.restore_snapshot(path)
            self._log("快照已整体替换：" + json.dumps(result, ensure_ascii=False, default=str))
        except WorkbenchError as exc:
            self._error(exc)
        finally:
            self._refresh_all()

    def _verify_replay(self) -> None:
        wb = self._require()
        if wb is None:
            return
        try:
            report = wb.replay_conclusions(
                repeats=2,
                timeout_ms=int(self.timeout_ms.get()),
                budget_ms=max(8000, int(self.budget_ms.get()) * 2),
            )
            self._log("重放校验结果：" + json.dumps(report, ensure_ascii=False, indent=2, default=str))
            messagebox.showinfo("重放校验", "一致" if report["matched"] else "存在差异，请查看输出")
        except (ValueError, WorkbenchError) as exc:
            if isinstance(exc, WorkbenchError):
                self._error(exc)
            else:
                messagebox.showerror("参数错误", "超时和预算必须是整数")

    def _on_close(self) -> None:
        try:
            if self.wb is not None:
                self.wb.close()
        finally:
            self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def run_gui(args: Any = None) -> int:
    initial = getattr(args, "session", None)
    samples = getattr(args, "samples", str(DEFAULT_SAMPLES_DIR))
    app = WorkbenchApp(initial_session=initial, samples_dir=samples)
    return app.run()
