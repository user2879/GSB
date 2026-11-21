"""Tkinter diagnostic workbench with a lazy, headless-safe import boundary."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import MemDiagError
from .ledger import Ledger
from .plan import apply_plan_changes, load_plan, load_samples, plan_from_dict, sample_set_from_dict
from .services import recover_session, replay_session, resume_session, run_full_session, run_incremental_session
from .snapshots import SnapshotStore
from .viewmodel import build_view_model, format_layer_chain


class GuiUnavailable(RuntimeError):
    pass


def run_gui(
    plan_path: str | Path | None = None,
    samples_path: str | Path | None = None,
    db_path: str | Path = "workbench.sqlite3",
    snapshot_dir: str | Path = "snapshots",
) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as error:  # pragma: no cover - depends on system Tk package
        raise GuiUnavailable(
            "Tkinter is unavailable in this Python installation. "
            "Install the platform Tk package (Windows Python includes it), then retry."
        ) from error

    palette = [
        "#4f7cac",
        "#d17b49",
        "#5b9b72",
        "#9b6bb2",
        "#c05067",
        "#7a8b45",
        "#3f9ca6",
        "#b08b3e",
        "#6f79c5",
        "#a35d8f",
    ]

    class Workbench(ttk.Frame):
        def __init__(self, root: "tk.Tk") -> None:
            super().__init__(root, padding=8)
            self.root = root
            self.root.title("混合精度推理显存预算诊断工作台")
            self.root.geometry("1320x860")
            self.root.minsize(1080, 700)
            self.ledger = Ledger(db_path)
            self.snapshot_store = SnapshotStore(snapshot_dir)
            self.plan_path = Path(plan_path) if plan_path else None
            self.samples_path = Path(samples_path) if samples_path else None
            self.current_run = None
            self.view = None
            self.layer_color = {}
            self._build_widgets()
            self._load_initial()

        def _build_widgets(self) -> None:
            self.pack(fill="both", expand=True)
            toolbar = ttk.Frame(self)
            toolbar.pack(fill="x", pady=(0, 6))
            buttons = [
                ("加载计划", self._choose_plan),
                ("加载样本", self._choose_samples),
                ("全量诊断", self._run_full),
                ("应用增量", self._run_incremental),
                ("保存快照", self._save_snapshot),
                ("恢复选中快照", self._restore_snapshot),
                ("扫描会话", self._scan_session),
                ("恢复并重放", self._resume_session),
            ]
            for label, command in buttons:
                ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=(0, 5))
            self.header_var = tk.StringVar(value="尚未运行")
            ttk.Label(toolbar, textvariable=self.header_var, anchor="e").pack(
                side="right", fill="x", expand=True
            )

            vertical = ttk.Panedwindow(self, orient="vertical")
            vertical.pack(fill="both", expand=True)
            upper = ttk.Panedwindow(vertical, orient="horizontal")
            vertical.add(upper, weight=4)

            left = ttk.Frame(upper, padding=(0, 0, 5, 0))
            upper.add(left, weight=3)
            ttk.Label(left, text="层依赖 / 失效标记 / 块序", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            columns = ("layer", "mode", "scale", "version", "invalid", "blocks", "error")
            self.layer_tree = ttk.Treeview(left, columns=columns, show="headings", height=13)
            headings = {
                "layer": "层标识",
                "mode": "精度",
                "scale": "缩放",
                "version": "版本",
                "invalid": "失效",
                "blocks": "块序号",
                "error": "误差界",
            }
            widths = {
                "layer": 130,
                "mode": 80,
                "scale": 55,
                "version": 55,
                "invalid": 55,
                "blocks": 160,
                "error": 90,
            }
            for column in columns:
                self.layer_tree.heading(column, text=headings[column])
                self.layer_tree.column(column, width=widths[column], anchor="w", stretch=column in {"layer", "blocks"})
            self.layer_tree.pack(fill="both", expand=True)
            self.layer_tree.bind("<<TreeviewSelect>>", self._on_layer_select)

            detail_frame = ttk.Frame(left)
            detail_frame.pack(fill="both", expand=True, pady=(5, 0))
            ttk.Label(detail_frame, text="归因链 / 引用 / 块布局", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            self.detail_text = tk.Text(detail_frame, height=13, wrap="word", relief="solid", borderwidth=1)
            self.detail_text.pack(fill="both", expand=True)
            self.detail_text.configure(state="disabled")

            center = ttk.Frame(upper, padding=(5, 0, 5, 0))
            upper.add(center, weight=4)
            notebook = ttk.Notebook(center)
            notebook.pack(fill="both", expand=True)
            block_tab = ttk.Frame(notebook, padding=5)
            error_tab = ttk.Frame(notebook, padding=5)
            batch_tab = ttk.Frame(notebook, padding=5)
            notebook.add(block_tab, text="块池占用")
            notebook.add(error_tab, text="逐层误差与归因")
            notebook.add(batch_tab, text="批次结论")

            memory_line = ttk.Frame(block_tab)
            memory_line.pack(fill="x")
            self.memory_var = tk.StringVar(value="无显存状态")
            ttk.Label(memory_line, textvariable=self.memory_var).pack(anchor="w")
            self.block_canvas = tk.Canvas(block_tab, background="#f4f4f4", highlightthickness=0, height=390)
            self.block_canvas.pack(fill="both", expand=True, pady=(5, 0))
            self.block_canvas.bind("<Configure>", lambda event: self._draw_blocks())
            self.block_legend = tk.Text(block_tab, height=4, wrap="word", relief="flat")
            self.block_legend.pack(fill="x", pady=(4, 0))
            self.block_legend.configure(state="disabled")

            error_columns = ("rank", "layer", "actual", "bound", "cumulative", "dominant", "tie")
            self.error_tree = ttk.Treeview(error_tab, columns=error_columns, show="headings")
            error_headings = {
                "rank": "排名",
                "layer": "层",
                "actual": "最大偏差",
                "bound": "误差界",
                "cumulative": "累计占比",
                "dominant": "主导段",
                "tie": "受误差影响",
            }
            for column in error_columns:
                self.error_tree.heading(column, text=error_headings[column])
                self.error_tree.column(column, width=110, anchor="w")
            self.error_tree.pack(fill="both", expand=True)

            batch_columns = ("batch", "samples", "blocks", "request", "peak", "frag")
            self.batch_tree = ttk.Treeview(batch_tab, columns=batch_columns, show="headings")
            for column, label in {
                "batch": "批次",
                "samples": "样本",
                "blocks": "块集合",
                "request": "请求字节",
                "peak": "峰值块",
                "frag": "碎片块",
            }.items():
                self.batch_tree.heading(column, text=label)
                self.batch_tree.column(column, width=140, anchor="w")
            self.batch_tree.pack(fill="both", expand=True)

            right = ttk.Frame(upper, padding=(5, 0, 0, 0))
            upper.add(right, weight=2)
            ttk.Label(right, text="计划版本 / 快照 / 账本会话", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            self.snapshot_tree = ttk.Treeview(
                right,
                columns=("snapshot", "version", "digest", "readonly"),
                show="headings",
                height=10,
            )
            for column, label in {
                "snapshot": "快照",
                "version": "计划版本",
                "digest": "内容摘要",
                "readonly": "只读",
            }.items():
                self.snapshot_tree.heading(column, text=label)
                self.snapshot_tree.column(column, width=120, anchor="w")
            self.snapshot_tree.pack(fill="both", expand=True)
            ttk.Label(right, text="会话恢复状态", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(8, 0))
            self.session_tree = ttk.Treeview(
                right,
                columns=("session", "status", "version", "committed"),
                show="headings",
                height=7,
            )
            for column, label in {
                "session": "会话",
                "status": "状态",
                "version": "版本",
                "committed": "已提交/总数",
            }.items():
                self.session_tree.heading(column, text=label)
                self.session_tree.column(column, width=110, anchor="w")
            self.session_tree.pack(fill="both", expand=True)
            self.recovery_text = tk.Text(right, height=9, wrap="word", relief="solid", borderwidth=1)
            self.recovery_text.pack(fill="both", expand=True, pady=(6, 0))
            self.recovery_text.configure(state="disabled")

            bottom = ttk.Frame(vertical, padding=(0, 5, 0, 0))
            vertical.add(bottom, weight=1)
            ttk.Label(bottom, text="稳定失效传播轨迹", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            trace_columns = ("seq", "layer", "source", "via", "depth", "reason", "path")
            self.trace_tree = ttk.Treeview(bottom, columns=trace_columns, show="headings", height=6)
            for column, label in {
                "seq": "序",
                "layer": "失效层",
                "source": "触发源",
                "via": "直接上游",
                "depth": "深度",
                "reason": "依据",
                "path": "路径",
            }.items():
                self.trace_tree.heading(column, text=label)
                self.trace_tree.column(column, width=130, anchor="w")
            self.trace_tree.pack(fill="both", expand=True)

            self.status_var = tk.StringVar(value="就绪")
            ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", pady=(6, 0))

        def _load_initial(self) -> None:
            if self.plan_path and self.samples_path:
                self._run_full()

        def _choose_plan(self) -> None:
            selected = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*")])
            if selected:
                self.plan_path = Path(selected)
                self.status_var.set(f"计划: {selected}")

        def _choose_samples(self) -> None:
            selected = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*")])
            if selected:
                self.samples_path = Path(selected)
                self.status_var.set(f"样本: {selected}")

        def _require_inputs(self):
            if not self.plan_path or not self.samples_path:
                messagebox.showinfo("缺少输入", "请先选择本地 JSON 计划与样本文件。")
                return None
            try:
                return load_plan(self.plan_path), load_samples(self.samples_path)
            except MemDiagError as error:
                messagebox.showerror(error.code, json.dumps(error.to_dict(), ensure_ascii=False, indent=2))
                self.status_var.set(f"{error.code}: {error.message}")
                return None

        def _run_full(self) -> None:
            inputs = self._require_inputs()
            if inputs is None:
                return
            plan, samples = inputs
            try:
                result = run_full_session(
                    plan,
                    samples,
                    self.ledger,
                    snapshot_store=self.snapshot_store,
                    save_snapshot=True,
                    snapshot_label="gui-full",
                )
                self.current_run = result["run"]
                self._refresh_view()
                self.status_var.set(f"全量诊断完成，会话 {result['session_id']}")
            except MemDiagError as error:
                messagebox.showerror(error.code, json.dumps(error.to_dict(), ensure_ascii=False, indent=2))
                self.status_var.set(f"{error.code}: {error.message}")

        def _run_incremental(self) -> None:
            if self.current_run is None:
                messagebox.showinfo("需要基线", "请先执行一次全量诊断或恢复快照。")
                return
            dialog = tk.Toplevel(self.root)
            dialog.title("增量修改 JSON")
            dialog.geometry("620x420")
            text = tk.Text(dialog, wrap="none")
            text.pack(fill="both", expand=True)
            example = {
                "layer_updates": {"b_conv": {"precision": "float32", "scale": 0}},
                "consumer_updates": {},
                "memory_updates": {},
                "delete_layers": [],
            }
            text.insert("1.0", json.dumps(example, ensure_ascii=False, indent=2))
            button_row = ttk.Frame(dialog)
            button_row.pack(fill="x")

            def apply_text() -> None:
                try:
                    changes = json.loads(text.get("1.0", "end"))
                    new_plan, changed_sources, details = apply_plan_changes(self.current_run.plan, changes)
                    result = run_incremental_session(
                        new_plan,
                        self.current_run.samples,
                        self.current_run,
                        changed_sources,
                        self.ledger,
                        snapshot_store=self.snapshot_store,
                        save_snapshot=True,
                        snapshot_label="gui-incremental",
                    )
                    self.current_run = result["run"]
                    dialog.destroy()
                    self._refresh_view()
                    self.status_var.set(
                        f"增量完成；失效层 {', '.join(result['invalidated'])}；修改 {len(details)} 项"
                    )
                except MemDiagError as error:
                    messagebox.showerror(error.code, json.dumps(error.to_dict(), ensure_ascii=False, indent=2))
                except Exception as error:
                    messagebox.showerror("增量修改失败", str(error))

            ttk.Button(button_row, text="取消", command=dialog.destroy).pack(side="right", padx=4)
            ttk.Button(button_row, text="应用并计算", command=apply_text).pack(side="right", padx=4)

        def _save_snapshot(self) -> None:
            if self.current_run is None:
                return
            info = self.snapshot_store.save(self.current_run, label="gui-manual", ledger=self.ledger)
            self._refresh_sidebar()
            self.status_var.set(f"快照已保存并只读固化: {info.snapshot_id}")

        def _selected_snapshot_id(self) -> str | None:
            selection = self.snapshot_tree.selection()
            return None if not selection else str(selection[0])

        def _restore_snapshot(self) -> None:
            snapshot_id = self._selected_snapshot_id()
            if not snapshot_id:
                messagebox.showinfo("选择快照", "请在右侧快照列表中选择一个快照。")
                return
            try:
                result = self.snapshot_store.restore(snapshot_id, self.ledger)
                restored = self.ledger.load_current_run()
                if restored is not None:
                    self.current_run = restored
                    self._refresh_view()
                self.status_var.set(
                    f"快照恢复完成；整体替换={result['replaced']}；备份={None if result['backup_snapshot'] is None else result['backup_snapshot']['snapshot_id']}"
                )
            except MemDiagError as error:
                messagebox.showerror(error.code, json.dumps(error.to_dict(), ensure_ascii=False, indent=2))

        def _selected_session_id(self) -> str | None:
            selection = self.session_tree.selection()
            return None if not selection else str(selection[0])

        def _scan_session(self) -> None:
            session_id = self._selected_session_id()
            if not session_id:
                messagebox.showinfo("选择会话", "请选择账本会话。")
                return
            try:
                report = recover_session(self.ledger, session_id)
                self._set_recovery_text(report)
                self.status_var.set(
                    "会话已完整提交" if report["exit_report_available"] else "存在未知未提交批次，必须重跑"
                )
                self._refresh_sidebar()
            except MemDiagError as error:
                messagebox.showerror(error.code, json.dumps(error.to_dict(), ensure_ascii=False, indent=2))

        def _resume_session(self) -> None:
            session_id = self._selected_session_id()
            if not session_id:
                messagebox.showinfo("选择会话", "请选择账本会话。")
                return
            try:
                report = resume_session(
                    self.ledger,
                    session_id,
                    snapshot_store=self.snapshot_store,
                    save_snapshot=True,
                )
                self._set_recovery_text(report)
                if "run" in report:
                    self.current_run = report["run"]
                    self._refresh_view()
                replay = replay_session(self.ledger, session_id)
                self.status_var.set(f"恢复与重放通过；核验层 {', '.join(replay['checked_layers'])}")
            except MemDiagError as error:
                messagebox.showerror(error.code, json.dumps(error.to_dict(), ensure_ascii=False, indent=2))

        def _refresh_view(self) -> None:
            if self.current_run is None:
                return
            snapshots = self.snapshot_store.list()
            sessions = self.ledger.list_sessions()
            self.view = build_view_model(self.current_run, snapshots=snapshots, sessions=sessions)
            self.layer_color = {
                layer["layer_id"]: palette[index % len(palette)]
                for index, layer in enumerate(self.view["layers"])
            }
            self.header_var.set(
                f"计划 v{self.view['header']['plan_version']}  |  摘要 {self.view['header']['content_digest'][:16]}  |  样本 {self.view['header']['sample_count']}  |  快照 {len(self.view['snapshots'])}"
            )
            for tree in (self.layer_tree, self.error_tree, self.batch_tree, self.trace_tree):
                tree.delete(*tree.get_children())
            for layer in self.view["layers"]:
                self.layer_tree.insert(
                    "",
                    "end",
                    iid=layer["layer_id"],
                    values=(
                        layer["layer_id"],
                        layer["precision"],
                        layer["scale"],
                        layer["version"],
                        "是" if layer["invalidated"] else "否",
                        ",".join(str(block) for block in layer["block_ids"]),
                        f"{layer['error_bound']:.8g}",
                    ),
                )
            for entry in self.view["error_chain"]:
                self.error_tree.insert(
                    "",
                    "end",
                    values=(
                        entry["rank"],
                        entry["layer_id"],
                        f"{entry['max_abs_error']:.8g}",
                        f"{entry['error_bound']:.8g}",
                        f"{entry['cumulative_error_ratio']:.3%}",
                        entry["dominant_segment"],
                        "是" if entry["tie_affected"] else "否",
                    ),
                )
            for batch in self.view["batches"]:
                self.batch_tree.insert(
                    "",
                    "end",
                    iid=batch["batch_id"],
                    values=(
                        batch["batch_id"],
                        ",".join(batch["sample_ids"]),
                        ",".join(str(block) for block in batch["block_ids"]),
                        batch["request_bytes"],
                        batch["peak_blocks"],
                        batch["fragmentation_blocks"],
                    ),
                )
            for index, step in enumerate(self.view["invalidation_trace"]):
                self.trace_tree.insert(
                    "",
                    "end",
                    values=(
                        index,
                        step["layer_id"],
                        step["trigger_source"],
                        step["via"] or "-",
                        step["depth"],
                        step["reason"],
                        " > ".join(step["path"]),
                    ),
                )
            memory = self.view["memory"]
            self.memory_var.set(
                f"块池 {memory['occupied_blocks']}/{memory['capacity_blocks']} 已用；空闲 {memory['free_blocks']}；最大连续空闲 {memory['largest_contiguous_free_blocks']}；碎片 {memory['fragmentation_blocks']} 块；峰值 {memory['peak_blocks']}；块大小 {memory['block_size_kb']} KB；预算 {memory['budget_mib']} MiB"
            )
            self._draw_blocks()
            self._refresh_sidebar()
            if self.view["layers"]:
                first = self.view["layers"][0]["layer_id"]
                self.layer_tree.selection_set(first)
                self._show_layer(first)

        def _refresh_sidebar(self) -> None:
            self.snapshot_tree.delete(*self.snapshot_tree.get_children())
            snapshots = self.snapshot_store.list()
            for info in sorted(snapshots, key=lambda item: (item.plan_version, item.snapshot_id)):
                self.snapshot_tree.insert(
                    "",
                    "end",
                    iid=info.snapshot_id,
                    values=(info.snapshot_id, info.plan_version, info.content_digest[:12], "是" if info.read_only else "否"),
                )
            self.session_tree.delete(*self.session_tree.get_children())
            for session in sorted(self.ledger.list_sessions(), key=lambda item: str(item["session_id"])):
                self.session_tree.insert(
                    "",
                    "end",
                    iid=session["session_id"],
                    values=(
                        session["session_id"],
                        session["status"],
                        session["plan_version"],
                        f"{session['completed_layers']}/{session['total_layers']}",
                    ),
                )
            if self.view is not None and self.view.get("snapshots"):
                self.header_var.set(
                    f"计划 v{self.view['header']['plan_version']}  |  摘要 {self.view['header']['content_digest'][:16]}  |  样本 {self.view['header']['sample_count']}  |  快照 {len(self.view['snapshots'])}"
                )

        def _draw_blocks(self) -> None:
            self.block_canvas.delete("all")
            if self.view is None:
                return
            blocks = self.view["blocks"]
            if not blocks:
                return
            width = max(300, self.block_canvas.winfo_width())
            columns = min(32, max(8, int(math.sqrt(len(blocks)) * 2)))
            gap = 2
            cell = max(5, min(22, (width - 8 - gap * columns) // columns))
            for index, block in enumerate(blocks):
                row = index // columns
                column = index % columns
                x1 = 4 + column * (cell + gap)
                y1 = 4 + row * (cell + gap)
                fill = self.layer_color.get(block["owner"], "#d2d2d2") if block["owner"] else "#eeeeee"
                outline = "#888888" if block["owner"] else "#cccccc"
                self.block_canvas.create_rectangle(
                    x1,
                    y1,
                    x1 + cell,
                    y1 + cell,
                    fill=fill,
                    outline=outline,
                    tags=(f"block:{index}",),
                )
            legend_lines = [
                f"{layer['layer_id']}: {','.join(str(block) for block in layer['block_ids'])}"
                for layer in self.view["layers"]
            ]
            self.block_legend.configure(state="normal")
            self.block_legend.delete("1.0", "end")
            self.block_legend.insert("1.0", "  |  ".join(legend_lines))
            self.block_legend.configure(state="disabled")

        def _on_layer_select(self, event: Any = None) -> None:
            selection = self.layer_tree.selection()
            if selection:
                self._show_layer(str(selection[0]))

        def _show_layer(self, layer_id: str) -> None:
            if self.view is None:
                return
            layer = next(item for item in self.view["layers"] if item["layer_id"] == layer_id)
            trace = layer.get("invalidation") or {}
            blocks = ",".join(str(block) for block in layer["block_ids"])
            components = layer["error_components"]
            text = "\n".join(
                [
                    format_layer_chain(layer),
                    f"版本: {layer['version']}    精度: {layer['precision']}    scale: {layer['scale']}",
                    f"输入/输出/参数: {layer['input_dim']} / {layer['output_dim']} / {layer['param_count']}",
                    f"失效: {'是' if layer['invalidated'] else '否'}；触发源: {trace.get('trigger_source', '-')}；依据: {trace.get('reason', '-')}",
                    f"传播路径: {' > '.join(trace.get('path', [])) or '-'}",
                    f"块号: {blocks}；模式: {layer['allocation_mode']}；请求: {layer['request_bytes']} bytes",
                    f"输出摘要: {layer['output_digest']}",
                    f"浮点摘要: {layer['float_digest']}",
                    f"最大偏差: {layer['max_abs_error']:.10g}；误差界: {layer['error_bound']:.10g}",
                    "三段误差界: " + ", ".join(f"{key}={components[key]:.10g}" for key in sorted(components)),
                    f"主导段: {layer['dominant_segment']}",
                ]
            )
            self.detail_text.configure(state="normal")
            self.detail_text.delete("1.0", "end")
            self.detail_text.insert("1.0", text)
            self.detail_text.configure(state="disabled")

        def _set_recovery_text(self, report: dict[str, Any]) -> None:
            concise = {
                key: value
                for key, value in report.items()
                if key not in {"run"}
            }
            self.recovery_text.configure(state="normal")
            self.recovery_text.delete("1.0", "end")
            self.recovery_text.insert("1.0", json.dumps(concise, ensure_ascii=False, indent=2))
            self.recovery_text.configure(state="disabled")

    try:
        root = tk.Tk()
    except Exception as error:  # pragma: no cover - depends on the host display
        raise GuiUnavailable(
            "Tk was found, but no interactive display could be opened. "
            "Set DISPLAY on Linux or run the command in a Windows desktop session."
        ) from error
    app = Workbench(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.ledger.close(), root.destroy()))
    root.mainloop()
