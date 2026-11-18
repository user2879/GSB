from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

from .codec import parse_document
from .config_workspace import ConfigWorkspace
from .derive import resolve_effective
from .errors import AppError, GuiError
from .events import load_events
from .ledger import Ledger, ReplayReport
from .logging_utils import StructuredLogger
from .models import (
    ConfigDocument,
    DECISION_ALLOW,
    DECISION_OVER_QUOTA,
    DECISION_RATE_LIMITED,
    Derivation,
    LIMIT_DIMENSIONS,
    VolumeConfig,
)
from .preview import PreviewResult, PreviewService

try:  # pragma: no cover - exercised on Windows, not the headless CI host
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # pragma: no cover
    tk = None
    ttk = None
    messagebox = None


DECISION_LABELS = {
    DECISION_ALLOW: "放行",
    DECISION_RATE_LIMITED: "限流",
    DECISION_OVER_QUOTA: "超配额",
}


if tk is not None:  # pragma: no cover - GUI requires a display

    class QuotaEditorApp:
        def __init__(
            self,
            root: Any,
            *,
            config_dir: str,
            events_path: str,
            ledger_path: str,
            logger: StructuredLogger,
            preview_budget_ms: int,
        ) -> None:
            self.root = root
            self.config_dir = config_dir
            self.events_path = events_path
            self.ledger_path = ledger_path
            self.logger = logger
            self.preview_budget_ms = preview_budget_ms

            self.workspace: ConfigWorkspace | None = None
            self.workspace_result: Any = None
            self.ledger: Ledger | None = None
            self.active: ConfigDocument | None = None
            self.events = []
            self.view: dict[str, Any] = {}
            self.volume_rows: dict[str, Any] = {}
            self.fast_preview: PreviewResult | None = None
            self.fast_preview_generation = -1
            self.preview_after_id: str | None = None
            self.preview_generation = 0
            self.worker_queue: queue.Queue[tuple[Any, ...]] = queue.Queue()

            self.root.title("存储配额与限流配置编辑器")
            self.root.geometry("1440x900")
            self.root.minsize(1100, 700)
            self._configure_style()
            self._build_ui()
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.root.after(60, self._poll_worker_queue)
            self._open_runtime()

        def _configure_style(self) -> None:
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("Treeview", rowheight=25)
            style.configure("Error.TLabel", foreground="#a00000")
            style.configure("Ok.TLabel", foreground="#126b30")
            style.configure("Stale.TLabel", foreground="#8a5a00")

        def _build_ui(self) -> None:
            toolbar = ttk.Frame(self.root, padding=(10, 8))
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="刷新配置/账本", command=self._open_runtime).pack(side="left")
            ttk.Button(toolbar, text="录入事件流", command=self._ingest_in_background).pack(
                side="left", padx=(8, 0)
            )
            ttk.Button(toolbar, text="权威重算", command=self._authoritative_preview).pack(
                side="left", padx=(8, 0)
            )
            ttk.Button(toolbar, text="恢复/重放校验", command=self._recover_in_background).pack(
                side="left", padx=(8, 0)
            )
            self.path_label = ttk.Label(
                toolbar,
                text=(
                    f"配置: {self.config_dir}  |  事件: {self.events_path}  |  "
                    f"SQLite: {self.ledger_path}"
                ),
            )
            self.path_label.pack(side="right")

            paned = ttk.Panedwindow(self.root, orient="horizontal")
            paned.pack(fill="both", expand=True, padx=10, pady=(0, 8))

            left = ttk.Frame(paned)
            right = ttk.Frame(paned)
            paned.add(left, weight=3)
            paned.add(right, weight=4)

            summary_frame = ttk.LabelFrame(left, text="卷列表 / 生效策略", padding=8)
            summary_frame.pack(fill="both", expand=True)
            columns = (
                "id",
                "format",
                "capacity",
                "rate",
                "burst",
                "dimension",
                "chain",
                "arbitration",
            )
            self.volume_tree = ttk.Treeview(
                summary_frame, columns=columns, show="headings", selectmode="browse", height=13
            )
            headings = {
                "id": "卷标识",
                "format": "格式版本",
                "capacity": "生效容量(字节)",
                "rate": "每秒上限",
                "burst": "突发桶",
                "dimension": "限流维度",
                "chain": "派生链",
                "arbitration": "仲裁依据",
            }
            widths = {
                "id": 100,
                "format": 110,
                "capacity": 120,
                "rate": 85,
                "burst": 80,
                "dimension": 90,
                "chain": 160,
                "arbitration": 240,
            }
            for column in columns:
                self.volume_tree.heading(column, text=headings[column])
                self.volume_tree.column(column, width=widths[column], stretch=True, anchor="w")
            volume_scroll = ttk.Scrollbar(
                summary_frame, orient="vertical", command=self.volume_tree.yview
            )
            self.volume_tree.configure(yscrollcommand=volume_scroll.set)
            self.volume_tree.pack(side="left", fill="both", expand=True)
            volume_scroll.pack(side="right", fill="y")
            self.volume_tree.bind("<<TreeviewSelect>>", self._on_volume_selected)

            editor = ttk.LabelFrame(left, text="编辑 / 快速预览", padding=8)
            editor.pack(fill="x", pady=(8, 0))
            form = ttk.Frame(editor)
            form.pack(fill="x")
            self.var_capacity = tk.StringVar()
            self.var_rate = tk.StringVar()
            self.var_burst = tk.StringVar()
            self.var_dimension = tk.StringVar(value=LIMIT_DIMENSIONS[0])
            self.var_upstream = tk.StringVar()
            self.var_factor = tk.StringVar()
            self.var_offset = tk.StringVar()

            fields = [
                ("容量上限字节", self.var_capacity),
                ("每秒上限", self.var_rate),
                ("突发桶容量", self.var_burst),
                ("上游卷", self.var_upstream),
                ("整数系数", self.var_factor),
                ("整数偏移", self.var_offset),
            ]
            for index, (label, variable) in enumerate(fields):
                row = index // 3
                column = (index % 3) * 2
                ttk.Label(form, text=label).grid(row=row, column=column, sticky="w", padx=(0, 5), pady=3)
                entry = ttk.Entry(form, textvariable=variable, width=16)
                entry.grid(row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=3)
                entry.bind("<KeyRelease>", self._schedule_fast_preview)
            ttk.Label(form, text="限流维度").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=3)
            dimension = ttk.Combobox(
                form,
                textvariable=self.var_dimension,
                values=LIMIT_DIMENSIONS,
                state="readonly",
                width=14,
            )
            dimension.grid(row=2, column=1, sticky="ew", padx=(0, 14), pady=3)
            dimension.bind("<<ComboboxSelected>>", self._schedule_fast_preview)
            ttk.Button(form, text="应用并整体替换当前视图", command=self._apply_edit).grid(
                row=2, column=4, columnspan=2, sticky="ew", pady=3
            )
            for column in (1, 3, 5):
                form.columnconfigure(column, weight=1)

            self.preview_status = ttk.Label(editor, text="尚未预览", style="Stale.TLabel")
            self.preview_status.pack(anchor="w", pady=(6, 0))
            self.error_bound_label = ttk.Label(
                editor,
                text="量化误差上界：单事件 0，聚合 0（整数/千分单位精确计算）",
            )
            self.error_bound_label.pack(anchor="w", pady=(2, 0))
            self.stale_label = ttk.Label(editor, text="陈旧状态：当前配置为权威视图")
            self.stale_label.pack(anchor="w", pady=(2, 0))

            notebook = ttk.Notebook(right)
            notebook.pack(fill="both", expand=True)

            events_tab = ttk.Frame(notebook)
            curve_tab = ttk.Frame(notebook)
            history_tab = ttk.Frame(notebook)
            errors_tab = ttk.Frame(notebook)
            notebook.add(events_tab, text="逐事件判定")
            notebook.add(curve_tab, text="用量曲线")
            notebook.add(history_tab, text="版本 / 陈旧标记")
            notebook.add(errors_tab, text="结构化错误")

            event_columns = (
                "seq",
                "timestamp",
                "volume",
                "decision",
                "bytes",
                "ops",
                "cumulative",
                "tokens",
                "error",
            )
            self.event_tree = ttk.Treeview(
                events_tab, columns=event_columns, show="headings", selectmode="browse"
            )
            event_headings = {
                "seq": "序号",
                "timestamp": "时间(ms)",
                "volume": "卷",
                "decision": "判定",
                "bytes": "写入字节",
                "ops": "操作数",
                "cumulative": "累计字节",
                "tokens": "令牌(千分)",
                "error": "误差界",
            }
            event_widths = {
                "seq": 55,
                "timestamp": 80,
                "volume": 85,
                "decision": 85,
                "bytes": 80,
                "ops": 65,
                "cumulative": 90,
                "tokens": 90,
                "error": 70,
            }
            for column in event_columns:
                self.event_tree.heading(column, text=event_headings[column])
                self.event_tree.column(
                    column, width=event_widths[column], stretch=(column in {"volume", "decision"})
                )
            event_y = ttk.Scrollbar(events_tab, orient="vertical", command=self.event_tree.yview)
            event_x = ttk.Scrollbar(events_tab, orient="horizontal", command=self.event_tree.xview)
            self.event_tree.configure(yscrollcommand=event_y.set, xscrollcommand=event_x.set)
            self.event_tree.grid(row=0, column=0, sticky="nsew")
            event_y.grid(row=0, column=1, sticky="ns")
            event_x.grid(row=1, column=0, sticky="ew")
            events_tab.rowconfigure(0, weight=1)
            events_tab.columnconfigure(0, weight=1)

            curve_controls = ttk.Frame(curve_tab, padding=8)
            curve_controls.pack(fill="x")
            ttk.Label(curve_controls, text="曲线卷").pack(side="left")
            self.curve_volume = ttk.Combobox(curve_controls, state="readonly", width=24)
            self.curve_volume.pack(side="left", padx=8)
            self.curve_volume.bind("<<ComboboxSelected>>", lambda event: self._draw_curve())
            self.curve_canvas = tk.Canvas(curve_tab, background="#fbfbf8", highlightthickness=0)
            self.curve_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.curve_canvas.bind("<Configure>", lambda event: self._draw_curve())

            history_columns = ("revision", "name", "schema", "stale", "created", "source")
            self.history_tree = ttk.Treeview(
                history_tab, columns=history_columns, show="headings", selectmode="browse"
            )
            history_headings = {
                "revision": "修订",
                "name": "名称",
                "schema": "格式",
                "stale": "陈旧标记",
                "created": "写入时间",
                "source": "来源",
            }
            for column in history_columns:
                self.history_tree.heading(column, text=history_headings[column])
                self.history_tree.column(column, width=130 if column == "source" else 90, stretch=True)
            self.history_tree.pack(fill="both", expand=True, padx=8, pady=8)

            error_columns = ("code", "operation", "volume", "field", "message", "time")
            self.error_tree = ttk.Treeview(
                errors_tab, columns=error_columns, show="headings", selectmode="browse"
            )
            for column in error_columns:
                self.error_tree.heading(column, text=column)
                self.error_tree.column(column, width=130 if column != "message" else 360, stretch=True)
            self.error_tree.pack(fill="both", expand=True, padx=8, pady=8)

            self.status_var = tk.StringVar(value="正在加载本地配置…")
            status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", padding=4)
            status.pack(fill="x", side="bottom")

        def _open_runtime(self) -> None:
            self._cancel_preview_timer()
            self.preview_generation += 1
            self.fast_preview = None
            self.fast_preview_generation = -1
            if self.ledger is not None:
                self.ledger.close()
                self.ledger = None
            try:
                self.ledger = Ledger(self.ledger_path)
                self.workspace = ConfigWorkspace(
                    self.config_dir,
                    checkpoint_callback=self.ledger.record_upgrade_checkpoint,
                )
                self.workspace_result = self.workspace.open()
                for error in self.workspace_result.errors:
                    self.logger.error(error, operation="gui.workspace")
                    if self.ledger is not None:
                        self.ledger.record_error(error, operation="gui.workspace")
                self.active = self.workspace_result.require_active()
                self.ledger.register_config(self.active)
                ledger_events = self.ledger.load_event_inputs()
                self.events = ledger_events if ledger_events else load_events(self.events_path)
                self.view = self.ledger.current_view()
                self._refresh_views()
                self.status_var.set(
                    f"已加载 {len(self.active.volumes)} 个卷；活动修订 {self.active.revision}；"
                    f"{len(self.view.get('judgments', []))} 条权威判定"
                )
            except AppError as exc:
                self.logger.error(exc, operation="gui.open")
                self.status_var.set(f"{exc.code}: {exc.message}")
                messagebox.showerror("加载失败", f"{exc.code}: {exc.message}")
                self.active = None

        def _refresh_views(self) -> None:
            if self.active is None:
                return
            effective = resolve_effective(self.active)
            selected = self._selected_volume_id()
            for item in self.volume_tree.get_children():
                self.volume_tree.delete(item)
            self.volume_rows = {}
            for volume_id in sorted(effective):
                volume = effective[volume_id]
                row = self.volume_tree.insert(
                    "",
                    "end",
                    values=(
                        volume_id,
                        f"v{self.active.schema_version} (源 v{self.active.original_version})",
                        volume.capacity_bytes,
                        volume.ops_per_sec,
                        volume.burst_capacity,
                        volume.limit_dimension,
                        " -> ".join(volume.derivation_chain),
                        f"{volume.arbitration}: {volume.capacity_basis}",
                    ),
                )
                self.volume_rows[volume_id] = row
            if selected in self.volume_rows:
                self.volume_tree.selection_set(self.volume_rows[selected])
            elif self.volume_rows:
                self.volume_tree.selection_set(next(iter(self.volume_rows.values())))
            self._load_form_from_selection()
            self._refresh_event_view()
            self._refresh_history_view()
            self._refresh_error_view()
            volumes = sorted({volume.volume_id for volume in self.active.volumes})
            self.curve_volume.configure(values=volumes)
            if not self.curve_volume.get() and volumes:
                self.curve_volume.set(volumes[0])
            self._draw_curve()

        def _refresh_event_view(self) -> None:
            for item in self.event_tree.get_children():
                self.event_tree.delete(item)
            for judgment in sorted(self.view.get("judgments", []), key=lambda row: row["seq"]):
                self.event_tree.insert(
                    "",
                    "end",
                    values=(
                        judgment["seq"],
                        judgment["timestamp_ms"],
                        judgment["volume_id"],
                        DECISION_LABELS.get(judgment["decision"], judgment["decision"]),
                        judgment["write_bytes"],
                        judgment["operation_count"],
                        judgment["cumulative_write_bytes"],
                        judgment["tokens_scaled"],
                        judgment["single_error_upper"],
                    ),
                )

        def _refresh_history_view(self) -> None:
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            for version in sorted(
                self.view.get("config_versions", []), key=lambda row: row["revision"]
            ):
                active = version["revision"] == self.view.get("active_revision") and not version["stale"]
                stale = "CURRENT" if active else ("STALE" if version["stale"] else "INACTIVE")
                self.history_tree.insert(
                    "",
                    "end",
                    tags=("stale",) if stale == "STALE" else (),
                    values=(
                        version["revision"],
                        version["name"],
                        f"v{version['schema_version']} (源 v{version['original_version']})",
                        stale,
                        version["created_at"],
                        version["source_path"],
                    ),
                )
            self.history_tree.tag_configure("stale", foreground="#8a5a00")
            history_count = sum(1 for row in self.view.get("config_versions", []) if row["stale"])
            self.stale_label.configure(
                text=(
                    f"陈旧状态：{history_count} 个历史修订标记为 STALE，不参与当前视图"
                    if history_count
                    else "陈旧状态：当前视图无陈旧行"
                )
            )

        def _refresh_error_view(self) -> None:
            for item in self.error_tree.get_children():
                self.error_tree.delete(item)
            rows = sorted(
                self.view.get("structured_events", []),
                key=lambda row: (row["created_at"], row.get("event_id", 0)),
            )
            for row in rows:
                self.error_tree.insert(
                    "",
                    "end",
                    values=(
                        row["code"],
                        row.get("operation") or "",
                        row.get("volume_id") or "",
                        row.get("field_path") or "",
                        row["message"],
                        row["created_at"],
                    ),
                )

        def _selected_volume_id(self) -> str | None:
            if not hasattr(self, "volume_tree"):
                return None
            selection = self.volume_tree.selection()
            if not selection:
                return None
            values = self.volume_tree.item(selection[0], "values")
            return str(values[0]) if values else None

        def _on_volume_selected(self, event: Any = None) -> None:
            self._load_form_from_selection()
            if self.curve_volume.get() != self._selected_volume_id():
                self.curve_volume.set(self._selected_volume_id() or "")
                self._draw_curve()

        def _load_form_from_selection(self) -> None:
            if self.active is None:
                return
            volume_id = self._selected_volume_id()
            if not volume_id:
                return
            volume = self.active.volume_map.get(volume_id)
            if volume is None:
                return
            self.var_capacity.set(str(volume.capacity_bytes))
            self.var_rate.set(str(volume.ops_per_sec))
            self.var_burst.set(str(volume.burst_capacity))
            self.var_dimension.set(volume.limit_dimension)
            if volume.derivation:
                self.var_upstream.set(volume.derivation.upstream_id)
                self.var_factor.set(str(volume.derivation.factor))
                self.var_offset.set(str(volume.derivation.offset))
            else:
                self.var_upstream.set("")
                self.var_factor.set("")
                self.var_offset.set("")

        def _parse_nonnegative(self, value: str, label: str) -> int:
            try:
                parsed = int(value.strip())
            except ValueError as exc:
                raise ValueError(f"{label}必须是整数") from exc
            if parsed < 0:
                raise ValueError(f"{label}不得为负")
            return parsed

        def _candidate_from_form(self) -> ConfigDocument:
            if self.active is None:
                raise ValueError("没有活动配置")
            volume_id = self._selected_volume_id()
            if not volume_id:
                raise ValueError("请先选择卷")
            existing = self.active.volume_map[volume_id]
            upstream = self.var_upstream.get().strip()
            if upstream:
                derivation = Derivation(
                    upstream_id=upstream,
                    factor=self._parse_nonnegative(self.var_factor.get(), "整数系数"),
                    offset=self._parse_nonnegative(self.var_offset.get(), "整数偏移"),
                )
            else:
                derivation = None
            updated = VolumeConfig(
                volume_id=volume_id,
                capacity_bytes=self._parse_nonnegative(self.var_capacity.get(), "容量上限"),
                ops_per_sec=self._parse_nonnegative(self.var_rate.get(), "每秒上限"),
                burst_capacity=self._parse_nonnegative(self.var_burst.get(), "突发桶容量"),
                limit_dimension=self.var_dimension.get() or existing.limit_dimension,
                derivation=derivation,
            )
            candidate = self.active.with_volume(updated)
            # Re-enter through the same codec used by files and scripts, so the
            # GUI cannot bypass validation or normalization.
            return parse_document(candidate.to_json(), source_path=self.active.source_path)

        def _schedule_fast_preview(self, event: Any = None) -> None:
            self.preview_generation += 1
            self.fast_preview = None
            self.fast_preview_generation = -1
            self._cancel_preview_timer()
            self.preview_after_id = self.root.after(180, self._run_fast_preview)

        def _cancel_preview_timer(self) -> None:
            if self.preview_after_id is not None:
                self.root.after_cancel(self.preview_after_id)
                self.preview_after_id = None

        def _run_fast_preview(self) -> None:
            self.preview_after_id = None
            generation = self.preview_generation
            try:
                candidate = self._candidate_from_form()
                effective = resolve_effective(candidate)
                volume_id = self._selected_volume_id()
                if volume_id and volume_id in effective:
                    volume = effective[volume_id]
                    self.preview_status.configure(
                        style="Stale.TLabel",
                        text=(
                            f"配置摘要：生效容量 {volume.capacity_bytes} 字节；"
                            f"仲裁 {volume.arbitration}；判定预览正在后台计算…"
                        ),
                    )
            except (ValueError, AppError) as exc:
                self.fast_preview = None
                self.fast_preview_generation = -1
                text = f"{exc.code}: {exc.message}" if isinstance(exc, AppError) else str(exc)
                self.preview_status.configure(text=f"预览不可用：{text}", style="Error.TLabel")
                if isinstance(exc, AppError):
                    self.logger.error(exc, operation="gui.preview.fast")
                return
            service = PreviewService(
                list(self.events), logger=self.logger, budget_ms=self.preview_budget_ms
            )

            def worker() -> None:
                try:
                    preview = service.fast_preview(candidate)
                    self.worker_queue.put(
                        ("fast_preview", generation, candidate, service, preview)
                    )
                except Exception as exc:  # thread boundary
                    self.worker_queue.put(("worker_error", generation, exc))

            threading.Thread(target=worker, daemon=True, name="fast-preview").start()

        def _show_fast_result(self, candidate: ConfigDocument, preview: PreviewResult) -> None:
            effective = resolve_effective(candidate)
            volume_id = self._selected_volume_id()
            if volume_id and volume_id in effective:
                volume = effective[volume_id]
                self.preview_status.configure(
                    style="Stale.TLabel" if preview.latency_exceeded else "Ok.TLabel",
                    text=(
                        f"快速预览：生效容量 {volume.capacity_bytes} 字节；"
                        f"仲裁 {volume.arbitration}；{preview.elapsed_ms}ms / "
                        f"预算 {preview.budget_ms}ms；"
                        f"放行 {preview.decision_counts.get(DECISION_ALLOW, 0)}，"
                        f"限流 {preview.decision_counts.get(DECISION_RATE_LIMITED, 0)}，"
                        f"超配额 {preview.decision_counts.get(DECISION_OVER_QUOTA, 0)}"
                    ),
                )
            self.error_bound_label.configure(
                text=(
                    "量化误差上界：单事件 0，聚合 0（整数/千分单位精确计算）"
                    f"；本预览 {preview.error_bound_single}/{preview.error_bound_aggregate}"
                )
            )

        def _authoritative_preview(self) -> None:
            try:
                candidate = self._candidate_from_form()
            except (ValueError, AppError) as exc:
                messagebox.showerror("预览失败", str(exc))
                return
            self._cancel_preview_timer()
            self.preview_generation += 1
            generation = self.preview_generation
            self.preview_status.configure(text="权威重算正在后台执行…", style="Stale.TLabel")

            def worker() -> None:
                try:
                    with Ledger(self.ledger_path) as ledger:
                        events = ledger.load_event_inputs()
                    if not events:
                        self.worker_queue.put(("no_ledger_events", generation))
                        return
                    service = PreviewService(
                        events, logger=self.logger, budget_ms=self.preview_budget_ms
                    )
                    preview = service.fast_preview(candidate)
                    authoritative = service.authoritative_replay(candidate)
                    comparison = service.compare(preview, authoritative)
                    self.worker_queue.put(
                        (
                            "authoritative",
                            generation,
                            candidate,
                            service,
                            preview,
                            authoritative,
                            comparison,
                        )
                    )
                except Exception as exc:  # thread boundary
                    self.worker_queue.put(("worker_error", generation, exc))

            threading.Thread(target=worker, daemon=True, name="authoritative-replay").start()

        def _apply_edit(self) -> None:
            if self.workspace is None or self.ledger is None or self.active is None:
                return
            try:
                self._cancel_preview_timer()
                self.preview_generation += 1
                self.fast_preview = None
                self.fast_preview_generation = -1
                changed = self._candidate_from_form()
                ledger_revision = self.ledger.active_revision() or 0
                revision = self.workspace.next_revision(minimum=ledger_revision)
                raw = changed.to_json()
                raw["revision"] = revision
                candidate = parse_document(raw, source_path=changed.source_path)
                resolve_effective(candidate)
                self.workspace.save_active(candidate, minimum_revision=ledger_revision)
                replaced = self.ledger.register_config(candidate)
                self.active = candidate
                self.workspace_result.active = candidate
                self.workspace_result.documents = list(self.workspace.documents)
                self.view = self.ledger.current_view()
                self._refresh_views()
                self.status_var.set(
                    f"修订 {revision} 已保存；当前视图已整体替换"
                    + ("；历史判定标记 STALE" if replaced else "")
                )
                self._ingest_in_background()
            except (ValueError, AppError) as exc:
                text = f"{exc.code}: {exc.message}" if isinstance(exc, AppError) else str(exc)
                if isinstance(exc, AppError):
                    self.logger.error(exc, operation="gui.apply")
                messagebox.showerror("应用失败", text)

        def _ingest_in_background(self) -> None:
            if self.active is None:
                return
            document = self.active
            events = list(self.events)
            generation = self.preview_generation

            def worker() -> None:
                try:
                    with Ledger(self.ledger_path) as led:
                        led.register_config(document)
                        ingest = led.ingest_events(document, events, batch_size=3)
                        replay = led.verify_replay()
                        self.worker_queue.put(("ingest_done", generation, ingest, replay))
                except Exception as exc:
                    self.worker_queue.put(("worker_error", generation, exc))

            self.status_var.set("正在按事件流重建权威账本…")
            threading.Thread(target=worker, daemon=True, name="ledger-ingest").start()

        def _recover_in_background(self) -> None:
            generation = self.preview_generation

            def worker() -> None:
                try:
                    with Ledger(self.ledger_path) as led:
                        recovery = led.recover_incomplete()
                        replay = led.verify_replay()
                        self.worker_queue.put(("recover_done", generation, recovery, replay))
                except Exception as exc:
                    self.worker_queue.put(("worker_error", generation, exc))

            self.status_var.set("正在恢复已提交批次并逐项重放校验…")
            threading.Thread(target=worker, daemon=True, name="ledger-recover").start()

        def _poll_worker_queue(self) -> None:
            try:
                while True:
                    item = self.worker_queue.get_nowait()
                    kind = item[0]
                    if kind == "fast_preview":
                        _, generation, candidate, service, preview = item
                        if generation != self.preview_generation:
                            continue
                        self.fast_preview = preview
                        self.fast_preview_generation = generation
                        self._show_fast_result(candidate, preview)
                    elif kind == "no_ledger_events":
                        _, generation = item
                        if generation != self.preview_generation:
                            continue
                        messagebox.showinfo("权威重算", "账本尚无已提交事件，请先录入事件流")
                        self.preview_status.configure(
                            text="权威重算等待账本事件输入", style="Stale.TLabel"
                        )
                    elif kind == "authoritative":
                        (
                            _,
                            generation,
                            candidate,
                            service,
                            preview,
                            authoritative,
                            comparison,
                        ) = item
                        if generation != self.preview_generation:
                            continue
                        self.fast_preview = preview
                        self.fast_preview_generation = generation
                        if comparison.ok:
                            style = "Stale.TLabel" if comparison.latency_exceeded else "Ok.TLabel"
                            self.preview_status.configure(
                                style=style,
                                text=(
                                    f"权威重放一致：{len(authoritative.judgments)} 个事件；"
                                    f"{authoritative.elapsed_ms}ms / 预算 {self.preview_budget_ms}ms；"
                                    f"误差上界 0"
                                ),
                            )
                        else:
                            first = comparison.mismatches[0]
                            self.preview_status.configure(
                                style="Error.TLabel",
                                text=(
                                    f"E_PREVIEW 卷 {first['volume_id']} 事件 {first['seq']} "
                                    f"差异 {','.join(first['difference_fields'])}"
                                ),
                            )
                    elif kind == "ingest_done":
                        _, generation, ingest, replay = item
                        self.view = self.ledger.current_view() if self.ledger else {}
                        self._refresh_event_view()
                        self._refresh_history_view()
                        self._refresh_error_view()
                        self._draw_curve()
                        replay_text = "重放一致" if replay.ok else "E_REPLAY"
                        self.status_var.set(
                            f"账本已提交 {ingest.batches_committed} 批、{ingest.judgments_written} 条判定；"
                            f"{replay_text}"
                        )
                        if not replay.ok and replay.errors:
                            self.logger.error(replay.errors[0], operation="gui.ingest.replay")
                    elif kind == "recover_done":
                        _, generation, recovery, replay = item
                        self.view = self.ledger.current_view() if self.ledger else {}
                        self._refresh_event_view()
                        self._refresh_history_view()
                        self._refresh_error_view()
                        self._draw_curve()
                        if recovery.rerun_required:
                            message = "存在未知批次，需要重跑；退出时报告不可用"
                        else:
                            message = "恢复完成；已提交批次重放一致" if replay.ok else "E_REPLAY"
                        self.status_var.set(message)
                    elif kind == "worker_error":
                        _, generation, exc = item
                        if isinstance(exc, ReplayReport):
                            text = "E_REPLAY"
                        else:
                            text = str(exc)
                        if isinstance(exc, AppError):
                            self.logger.error(exc, operation="gui.worker")
                        self.status_var.set(f"后台任务失败：{text}")
                        messagebox.showerror("后台任务失败", text)
            except queue.Empty:
                pass
            finally:
                self.root.after(60, self._poll_worker_queue)

        def _draw_curve(self) -> None:
            if not hasattr(self, "curve_canvas"):
                return
            canvas = self.curve_canvas
            canvas.delete("all")
            width = max(canvas.winfo_width(), 300)
            height = max(canvas.winfo_height(), 200)
            volume_id = self.curve_volume.get() or self._selected_volume_id()
            judgments = [
                row
                for row in self.view.get("judgments", [])
                if not volume_id or row["volume_id"] == volume_id
            ]
            judgments.sort(key=lambda row: row["seq"])
            if not judgments:
                canvas.create_text(
                    width / 2,
                    height / 2,
                    text="暂无权威判定数据；应用配置或录入事件流后将显示用量曲线",
                    fill="#666666",
                )
                return
            margin_left, margin_right, margin_top, margin_bottom = 52, 24, 36, 42
            plot_width = max(1, width - margin_left - margin_right)
            plot_height = max(1, height - margin_top - margin_bottom)
            max_bytes = max(1, max(row["cumulative_write_bytes"] for row in judgments))
            max_tokens = max(1, max(row["tokens_scaled"] for row in judgments))
            count = max(1, len(judgments) - 1)

            def x_for(index: int) -> float:
                return margin_left + (plot_width * index / count)

            def y_for(value: int, maximum: int) -> float:
                return margin_top + plot_height - (plot_height * value / maximum)

            canvas.create_line(
                margin_left,
                margin_top + plot_height,
                margin_left + plot_width,
                margin_top + plot_height,
                fill="#999999",
            )
            canvas.create_line(
                margin_left,
                margin_top,
                margin_left,
                margin_top + plot_height,
                fill="#999999",
            )
            canvas.create_text(8, 14, text=f"卷 {volume_id or '全部'}", anchor="w", fill="#333333")
            canvas.create_text(
                width - 12,
                14,
                text="蓝：累计写入字节  绿：令牌(千分)  红点：拒绝",
                anchor="e",
                fill="#555555",
            )
            byte_points: list[float] = []
            token_points: list[float] = []
            for index, row in enumerate(judgments):
                x = x_for(index)
                byte_points.extend((x, y_for(row["cumulative_write_bytes"], max_bytes)))
                token_points.extend((x, y_for(row["tokens_scaled"], max_tokens)))
                if row["decision"] != DECISION_ALLOW:
                    color = "#b00020" if row["decision"] == DECISION_OVER_QUOTA else "#d06b00"
                    canvas.create_oval(x - 3, y_for(row["cumulative_write_bytes"], max_bytes) - 3, x + 3, y_for(row["cumulative_write_bytes"], max_bytes) + 3, fill=color, outline=color)
                if index == 0 or index == len(judgments) - 1:
                    canvas.create_text(x, margin_top + plot_height + 14, text=str(row["seq"]), fill="#555555")
            if len(judgments) == 1:
                x = x_for(0)
                byte_points = [x, y_for(judgments[0]["cumulative_write_bytes"], max_bytes)]
                token_points = [x, y_for(judgments[0]["tokens_scaled"], max_tokens)]
            if len(byte_points) >= 4:
                canvas.create_line(*byte_points, fill="#1f5fa8", width=2, smooth=False)
            if len(token_points) >= 4:
                canvas.create_line(*token_points, fill="#17804a", width=2, smooth=False)
            canvas.create_text(
                margin_left,
                margin_top + plot_height + 30,
                text=f"累计字节峰值 {max_bytes}    令牌峰值 {max_tokens}/1000",
                anchor="w",
                fill="#555555",
            )

        def _on_close(self) -> None:
            if self.ledger is not None:
                self.ledger.close()
            self.root.destroy()


def launch_gui(
    *,
    config_dir: str,
    events_path: str,
    ledger_path: str,
    log_path: str,
    preview_budget_ms: int,
) -> None:
    if tk is None:  # pragma: no cover
        raise GuiError(
            "Tkinter is not available in this Python installation",
            details={"hint": "Install the Windows Python official distribution"},
        )
    logger = StructuredLogger(log_path)
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover
        raise GuiError(f"cannot create Tk window: {exc}") from exc
    QuotaEditorApp(
        root,
        config_dir=config_dir,
        events_path=events_path,
        ledger_path=ledger_path,
        logger=logger,
        preview_budget_ms=preview_budget_ms,
    )
    root.mainloop()
