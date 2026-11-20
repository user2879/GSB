from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from .canonical import fraction_decimal, write_json_atomic
from .errors import AppError
from .generator import generate_records
from .ledger import Ledger
from .logging_utils import JsonlLogger
from .replay import verify_ledger_run
from .spec import (
    PoliciesDoc,
    ServicesDoc,
    parse_policies_document,
    parse_services_document,
)
from .workflow import analyze_and_persist, recover_latest


def _fraction_text(value: Any, places: int = 3) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, dict):
            item = Fraction(int(value["numerator"]), int(value["denominator"]))
        else:
            item = Fraction(value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return str(value)
    return fraction_decimal(item, places)


def launch_gui(
    *,
    services_path: Path,
    policies_path: Path,
    db_path: Path,
    log_dir: Path,
) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable. On Windows, reinstall Python with the Tcl/Tk option "
            "and verify with: python -c \"import tkinter\""
        ) from exc

    class TraceDiagApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Trace Diag · 本地日志与追踪采样诊断")
            self.geometry("1480x920")
            self.minsize(1120, 720)
            self.services_path = services_path
            self.policies_path = policies_path
            self.db_path = db_path
            self.log_dir = log_dir
            self.logger = JsonlLogger(log_dir)
            self.view: dict[str, Any] | None = None
            self.services_doc: ServicesDoc | None = None
            self.policies_doc: PoliciesDoc | None = None
            self.records_by_service: dict[str, tuple[dict[str, Any], ...]] = {}
            self.selected_service: str | None = None
            self.selected_policy: str | None = None
            self.sample_page = 0
            self.sample_page_size = 50
            self._loading_policy_fields = False
            self._build_layout()
            self._load_initial()

        def _build_layout(self) -> None:
            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("Treeview", rowheight=25)
            style.configure("HeatLow.Treeview", background="#e8f5e9")
            style.configure("HeatMedium.Treeview", background="#fff8e1")
            style.configure("HeatHigh.Treeview", background="#ffebee")
            style.configure("TButton", padding=(8, 4))

            toolbar = ttk.Frame(self, padding=(10, 8))
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="刷新当前视图", command=self._refresh_view).pack(side="left")
            ttk.Button(toolbar, text="重算并整体替换", command=self._analyze_current).pack(
                side="left", padx=(6, 0)
            )
            ttk.Button(toolbar, text="恢复未完成批次", command=self._recover).pack(
                side="left", padx=(6, 0)
            )
            ttk.Button(toolbar, text="重放校验", command=self._verify).pack(
                side="left", padx=(6, 0)
            )
            self.status_var = tk.StringVar(value="正在读取账本…")
            ttk.Label(toolbar, textvariable=self.status_var).pack(side="left", padx=14)

            main = ttk.Panedwindow(self, orient="horizontal")
            main.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            left = ttk.Frame(main)
            right = ttk.Frame(main)
            main.add(left, weight=1)
            main.add(right, weight=4)

            ttk.Label(left, text="服务列表", font=("TkDefaultFont", 11, "bold")).pack(
                anchor="w", pady=(2, 5)
            )
            self.service_tree = ttk.Treeview(
                left,
                columns=("id", "policy", "p95", "bound"),
                show="headings",
                selectmode="browse",
                height=20,
            )
            for column, label, width in (
                ("id", "服务", 150),
                ("policy", "当前策略", 115),
                ("p95", "估计 p95 µs", 100),
                ("bound", "误差界 µs", 90),
            ):
                self.service_tree.heading(column, text=label)
                self.service_tree.column(column, width=width, anchor="w")
            self.service_tree.pack(fill="both", expand=True)
            self.service_tree.bind("<<TreeviewSelect>>", self._on_service_select)

            self.notebook = ttk.Notebook(right)
            self.notebook.pack(fill="both", expand=True)
            self._build_policy_tab()
            self._build_bucket_tab()
            self._build_attribution_tab()
            self._build_flip_tab()
            self._build_sample_tab()
            self._build_ledger_tab()

        def _build_policy_tab(self) -> None:
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text="策略与分位")
            top = ttk.Frame(frame)
            top.pack(fill="x")
            ttk.Label(top, text="候选策略").pack(side="left")
            self.policy_combo = ttk.Combobox(top, state="readonly", width=24)
            self.policy_combo.pack(side="left", padx=(6, 14))
            self.policy_combo.bind("<<ComboboxSelected>>", self._on_policy_select)
            self.version_label = ttk.Label(top, text="")
            self.version_label.pack(side="left")

            editor = ttk.LabelFrame(frame, text="策略参数编辑（应用后版本递增并整体替换）")
            editor.pack(fill="x", pady=(8, 8))
            labels = (
                ("head_ratio_bp", "头部比例(万分比)"),
                ("tail_duration_threshold_us", "尾部阈值(µs)"),
                ("tail_ratio_bp", "尾部比例(万分比)"),
                ("scale_power", "缩放幂(0-6)"),
                ("bucket_count", "分桶数"),
            )
            self.policy_entries: dict[str, Any] = {}
            for index, (key, label) in enumerate(labels):
                ttk.Label(editor, text=label).grid(row=0, column=index * 2, padx=(6, 2), pady=4)
                entry = ttk.Entry(editor, width=12)
                entry.grid(row=0, column=index * 2 + 1, padx=(0, 8), pady=4)
                self.policy_entries[key] = entry
            ttk.Label(editor, text="描述").grid(row=1, column=0, sticky="w", padx=6, pady=4)
            self.policy_description = ttk.Entry(editor, width=76)
            self.policy_description.grid(
                row=1, column=1, columnspan=7, sticky="ew", padx=(0, 8), pady=4
            )
            ttk.Button(editor, text="应用策略参数", command=self._apply_policy_edits).grid(
                row=1, column=8, padx=8, pady=4
            )
            editor.columnconfigure(7, weight=1)

            service_editor = ttk.LabelFrame(frame, text="服务描述编辑")
            service_editor.pack(fill="x", pady=(0, 8))
            self.service_description = ttk.Entry(service_editor)
            self.service_description.pack(side="left", fill="x", expand=True, padx=6, pady=5)
            ttk.Button(
                service_editor, text="保存描述并重算", command=self._apply_service_description
            ).pack(side="right", padx=6, pady=5)

            columns = (
                "percent",
                "full",
                "estimate",
                "diff",
                "sampling",
                "quant",
                "bucket",
                "total",
                "in_bound",
            )
            self.quantile_tree = ttk.Treeview(
                frame, columns=columns, show="headings", height=12
            )
            headings = (
                ("percent", "分位", 60),
                ("full", "全量定点 µs", 120),
                ("estimate", "采样估计 µs", 120),
                ("diff", "差值 µs", 90),
                ("sampling", "采样秩界 µs", 110),
                ("quant", "量化界 µs", 100),
                ("bucket", "桶边界界 µs", 110),
                ("total", "合成误差界 µs", 120),
                ("in_bound", "界内", 70),
            )
            for column, label, width in headings:
                self.quantile_tree.heading(column, text=label)
                self.quantile_tree.column(column, width=width, anchor="e")
            self.quantile_tree.tag_configure("low", background="#e8f5e9")
            self.quantile_tree.tag_configure("medium", background="#fff8e1")
            self.quantile_tree.tag_configure("high", background="#ffebee")
            self.quantile_tree.pack(fill="both", expand=True, pady=(4, 0))

        def _build_bucket_tab(self) -> None:
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text="逐桶统计")
            self.bucket_tree = ttk.Treeview(
                frame,
                columns=("index", "lower", "upper", "count", "hits", "head", "tail", "digest"),
                show="headings",
            )
            for column, label, width in (
                ("index", "桶", 50),
                ("lower", "下界 µs", 100),
                ("upper", "上界(不含) µs", 120),
                ("count", "记录数", 75),
                ("hits", "采样命中", 80),
                ("head", "头部命中", 80),
                ("tail", "尾部命中", 80),
                ("digest", "桶摘要", 280),
            ):
                self.bucket_tree.heading(column, text=label)
                self.bucket_tree.column(column, width=width, anchor="w")
            self.bucket_tree.pack(fill="both", expand=True)

        def _build_attribution_tab(self) -> None:
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text="归因贡献链")
            self.attribution_tree = ttk.Treeview(
                frame,
                columns=(
                    "target",
                    "count",
                    "hits",
                    "estimate",
                    "truth",
                    "diff",
                    "sparse",
                    "fixed",
                    "boundary",
                    "bound",
                    "dominant",
                ),
                show="headings",
            )
            for column, label, width in (
                ("target", "桶/总计", 95),
                ("count", "记录", 60),
                ("hits", "命中", 60),
                ("estimate", "估计 µs", 105),
                ("truth", "真实 µs", 105),
                ("diff", "差 µs", 95),
                ("sparse", "稀疏", 85),
                ("fixed", "定点", 85),
                ("boundary", "桶边界", 85),
                ("bound", "总界", 85),
                ("dominant", "主导段", 100),
            ):
                self.attribution_tree.heading(column, text=label)
                self.attribution_tree.column(column, width=width, anchor="e")
            self.attribution_tree.pack(fill="both", expand=True)

        def _build_flip_tab(self) -> None:
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text="翻转诊断")
            paned = ttk.Panedwindow(frame, orient="vertical")
            paned.pack(fill="both", expand=True)
            top = ttk.Frame(paned)
            bottom = ttk.Frame(paned)
            paned.add(top, weight=3)
            paned.add(bottom, weight=2)
            self.flip_tree = ttk.Treeview(
                top,
                columns=("kind", "subject", "parameter", "difference", "bound", "status"),
                show="headings",
            )
            for column, label, width in (
                ("kind", "类型", 140),
                ("subject", "对象", 260),
                ("parameter", "参数/对照", 220),
                ("difference", "差值 µs", 120),
                ("bound", "合成误差界 µs", 140),
                ("status", "状态", 150),
            ):
                self.flip_tree.heading(column, text=label)
                self.flip_tree.column(column, width=width, anchor="w")
            self.flip_tree.pack(fill="both", expand=True)
            self.flip_tree.bind("<<TreeviewSelect>>", self._on_flip_select)
            self.flip_detail = tk.Text(bottom, height=12, wrap="word")
            self.flip_detail.pack(fill="both", expand=True, pady=(6, 0))
            self._flip_payloads: dict[str, Any] = {}

        def _build_sample_tab(self) -> None:
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text="样本查看")
            controls = ttk.Frame(frame)
            controls.pack(fill="x", pady=(0, 6))
            ttk.Label(controls, text="过滤（客户端键/追踪标识）").pack(side="left")
            self.sample_filter = ttk.Entry(controls, width=30)
            self.sample_filter.pack(side="left", padx=6)
            ttk.Label(controls, text="排序").pack(side="left", padx=(12, 2))
            self.sample_sort = ttk.Combobox(
                controls,
                state="readonly",
                values=("record_index", "duration_us", "client_key"),
                width=16,
            )
            self.sample_sort.current(0)
            self.sample_sort.pack(side="left")
            ttk.Button(controls, text="应用", command=self._apply_sample_filter).pack(
                side="left", padx=6
            )
            ttk.Button(controls, text="上一页", command=lambda: self._page_samples(-1)).pack(
                side="left", padx=(18, 3)
            )
            ttk.Button(controls, text="下一页", command=lambda: self._page_samples(1)).pack(
                side="left"
            )
            self.sample_page_label = ttk.Label(controls, text="")
            self.sample_page_label.pack(side="left", padx=10)
            self.sample_tree = ttk.Treeview(
                frame,
                columns=(
                    "index",
                    "timestamp",
                    "magnitude",
                    "client",
                    "duration",
                    "trace",
                    "hint",
                    "decision",
                ),
                show="headings",
            )
            for column, label, width in (
                ("index", "序号", 65),
                ("timestamp", "时间戳 ms", 135),
                ("magnitude", "量级", 65),
                ("client", "客户端键", 150),
                ("duration", "耗时 µs", 100),
                ("trace", "追踪标识", 260),
                ("hint", "样本标记", 80),
                ("decision", "采样判定", 95),
            ):
                self.sample_tree.heading(column, text=label)
                self.sample_tree.column(column, width=width, anchor="w")
            self.sample_tree.pack(fill="both", expand=True)

        def _build_ledger_tab(self) -> None:
            frame = ttk.Frame(self.notebook, padding=8)
            self.notebook.add(frame, text="账本批次")
            ttk.Label(frame, text="全量 run 历史（stale 不参与当前展示）").pack(anchor="w")
            self.run_tree = ttk.Treeview(
                frame,
                columns=("run", "status", "active", "stale", "service", "strategy", "committed", "snapshot"),
                show="headings",
                height=8,
            )
            for column, label, width in (
                ("run", "Run", 230),
                ("status", "状态", 100),
                ("active", "活动", 60),
                ("stale", "陈旧", 60),
                ("service", "样本版本", 85),
                ("strategy", "策略版本", 85),
                ("committed", "批次", 80),
                ("snapshot", "内容摘要", 280),
            ):
                self.run_tree.heading(column, text=label)
                self.run_tree.column(column, width=width, anchor="w")
            self.run_tree.pack(fill="x", pady=(4, 10))
            ttk.Label(frame, text="当前 run 批次").pack(anchor="w")
            self.ledger_batch_tree = ttk.Treeview(
                frame,
                columns=("ordinal", "service", "status", "cursor", "digest"),
                show="headings",
            )
            for column, label, width in (
                ("ordinal", "序号", 60),
                ("service", "服务", 180),
                ("status", "状态", 100),
                ("cursor", "进度游标", 90),
                ("digest", "批次摘要", 360),
            ):
                self.ledger_batch_tree.heading(column, text=label)
                self.ledger_batch_tree.column(column, width=width, anchor="w")
            self.ledger_batch_tree.pack(fill="both", expand=True)

        def _load_initial(self) -> None:
            try:
                self._load_from_ledger()
            except Exception as exc:
                self._handle_error(exc)
                self._load_documents_from_files()

        def _load_documents_from_files(self) -> None:
            try:
                from .spec import load_policies_file, load_services_file

                self.services_doc = load_services_file(self.services_path)
                self.policies_doc = load_policies_file(self.policies_path)
                self.status_var.set("尚无活动账本；可点击“重算并整体替换”。")
            except Exception as exc:
                self._handle_error(exc)

        def _load_from_ledger(self) -> None:
            with Ledger(self.db_path) as ledger:
                view = ledger.get_run_view(verify=True)
                runs = ledger.list_runs()
                current_row = ledger.get_current_run_row()
                batch_run_id = current_row["run_id"] if current_row else (runs[0]["run_id"] if runs else None)
                batches = ledger.list_batches(batch_run_id) if batch_run_id else []
            self.run_tree.delete(*self.run_tree.get_children())
            for run in runs:
                self.run_tree.insert(
                    "",
                    "end",
                    values=(
                        run["run_id"],
                        run["status"],
                        "是" if run["active"] else "否",
                        "是" if run["stale"] else "否",
                        run["service_version"],
                        run["strategy_version"],
                        f"{run['committed_batches']}/{run['expected_batches']}",
                        run["snapshot_digest"] or "-",
                    ),
                )
            self.ledger_batch_tree.delete(*self.ledger_batch_tree.get_children())
            for batch in batches:
                self.ledger_batch_tree.insert(
                    "",
                    "end",
                    values=(
                        batch["ordinal"],
                        batch["service_id"],
                        batch["status"],
                        batch["progress_cursor"],
                        batch["content_digest"] or "-",
                    ),
                )
            if view is None:
                self.view = None
                self._load_documents_from_files()
                return
            self.view = view
            self.services_doc = parse_services_document(view["run"]["services"])
            self.policies_doc = parse_policies_document(view["run"]["policies"])
            self.records_by_service.clear()
            self._populate_policy_combo()
            self._populate_service_tree()
            self.status_var.set(
                f"活动 {view['run']['run_id']} · 样本 v{view['run']['service_version']} · "
                f"策略 v{view['run']['strategy_version']} · 批次 "
                f"{view['run']['committed_batches']}/{view['run']['expected_batches']}"
            )
            if self.service_tree.get_children():
                first = self.service_tree.get_children()[0]
                self.service_tree.selection_set(first)
                self.service_tree.focus(first)
                self._show_service(first)

        def _populate_policy_combo(self) -> None:
            if not self.policies_doc:
                return
            policy_ids = [policy.id for policy in self.policies_doc.policies]
            self.policy_combo["values"] = policy_ids
            if policy_ids:
                if self.selected_policy not in policy_ids:
                    self.selected_policy = policy_ids[0]
                self.policy_combo.set(self.selected_policy)
            self.version_label.configure(
                text=f"策略版本 v{self.policies_doc.version} · 定点粒度 10^p µs · 向下截断"
            )

        def _stream_summary_by_service(self) -> dict[str, dict[str, Any]]:
            if not self.view:
                return {}
            return {
                service["service_id"]: service["summary"]["stream_summary"]
                for service in self.view["services"]
                if service["summary"]
            }

        def _policy_choice(self) -> dict[str, Any]:
            selected = {}
            for service in self.view["services"] if self.view else []:
                summary = service["summary"]
                errors = service["error_bounds"]
                if not summary or not errors:
                    continue
                candidates = []
                for policy_id, row in errors["policies"].items():
                    mean = row["estimated_mean_us"]
                    candidates.append(
                        (
                            Fraction(mean["numerator"], mean["denominator"]),
                            row.get("policy_version", 0),
                            policy_id,
                        )
                    )
                winner = sorted(candidates)[0][2]
                selected[service["service_id"]] = winner
            return selected

        def _populate_service_tree(self) -> None:
            self.service_tree.delete(*self.service_tree.get_children())
            if not self.view:
                return
            winners = self._policy_choice()
            for service in sorted(self.view["services"], key=lambda item: item["ordinal"]):
                summary = service["summary"]
                errors = service["error_bounds"]
                if not summary or not errors:
                    continue
                policy_id = winners.get(service["service_id"], "")
                result = errors["policies"].get(policy_id, {})
                p95 = result.get("quantiles", {}).get("p95", {})
                bound = p95.get("total_error_bound_us")
                self.service_tree.insert(
                    "",
                    "end",
                    iid=service["service_id"],
                    values=(
                        service["service_id"],
                        policy_id,
                        p95.get("estimated_fixed_us", "-"),
                        _fraction_text(bound, 1),
                    ),
                )

        def _current_service_row(self) -> dict[str, Any] | None:
            if not self.view or not self.selected_service:
                return None
            for service in self.view["services"]:
                if service["service_id"] == self.selected_service:
                    return service
            return None

        def _policy_error_row(self) -> dict[str, Any] | None:
            service = self._current_service_row()
            if not service or not service["error_bounds"] or not self.selected_policy:
                return None
            return service["error_bounds"]["policies"].get(self.selected_policy)

        def _on_service_select(self, _event: Any = None) -> None:
            selection = self.service_tree.selection()
            if not selection:
                return
            self._show_service(selection[0])

        def _show_service(self, service_id: str) -> None:
            self.selected_service = service_id
            service = self._current_service_row()
            if service and service["summary"]:
                self.service_description.delete(0, "end")
                self.service_description.insert(0, service["summary"].get("description", ""))
                winners = self._policy_choice()
                if winners.get(service_id):
                    self.selected_policy = winners[service_id]
                    self.policy_combo.set(self.selected_policy)
            self.sample_page = 0
            self._refresh_policy_fields()
            self._refresh_buckets()
            self._refresh_quantiles()
            self._refresh_attribution()
            self._refresh_samples()

        def _on_policy_select(self, _event: Any = None) -> None:
            self.selected_policy = self.policy_combo.get()
            self._refresh_policy_fields()
            self._refresh_buckets()
            self._refresh_quantiles()
            self._refresh_attribution()
            self._refresh_samples()

        def _selected_policy_spec(self) -> Any | None:
            if not self.policies_doc or not self.selected_policy:
                return None
            return self.policies_doc.policy_map().get(self.selected_policy)

        def _refresh_policy_fields(self) -> None:
            policy = self._selected_policy_spec()
            if not policy:
                return
            self._loading_policy_fields = True
            try:
                self.policy_entries["head_ratio_bp"].delete(0, "end")
                self.policy_entries["head_ratio_bp"].insert(0, str(policy.head_ratio_bp))
                self.policy_entries["tail_duration_threshold_us"].delete(0, "end")
                self.policy_entries["tail_duration_threshold_us"].insert(
                    0, str(policy.tail_duration_threshold_us)
                )
                self.policy_entries["tail_ratio_bp"].delete(0, "end")
                self.policy_entries["tail_ratio_bp"].insert(0, str(policy.tail_ratio_bp))
                self.policy_entries["scale_power"].delete(0, "end")
                self.policy_entries["scale_power"].insert(0, str(policy.scale_power))
                self.policy_entries["bucket_count"].delete(0, "end")
                self.policy_entries["bucket_count"].insert(0, str(policy.bucket_count))
                self.policy_description.delete(0, "end")
                self.policy_description.insert(0, policy.description)
            finally:
                self._loading_policy_fields = False

        def _refresh_buckets(self) -> None:
            self.bucket_tree.delete(*self.bucket_tree.get_children())
            service = self._current_service_row()
            if not service:
                return
            rows = [
                bucket
                for bucket in service["buckets"]
                if bucket["policy_id"] == self.selected_policy
            ]
            for row in rows:
                self.bucket_tree.insert(
                    "",
                    "end",
                    values=(
                        row["bucket_index"],
                        row["lower_us"],
                        row["upper_us_exclusive"],
                        row["record_count"],
                        row["sample_hits"],
                        row["head_hits"],
                        row["tail_hits"],
                        row["bucket_digest"],
                    ),
                )

        def _refresh_quantiles(self) -> None:
            self.quantile_tree.delete(*self.quantile_tree.get_children())
            result = self._policy_error_row()
            if not result:
                return
            for key in ("p50", "p90", "p95", "p99"):
                row = result["quantiles"][key]
                total = Fraction(
                    row["total_error_bound_us"]["numerator"],
                    row["total_error_bound_us"]["denominator"],
                )
                ratio = total / max(Fraction(1), Fraction(row["full_fixed_us"]))
                tag = "low" if ratio < Fraction(5, 1000) else "medium" if ratio < Fraction(3, 100) else "high"
                self.quantile_tree.insert(
                    "",
                    "end",
                    values=(
                        key.upper(),
                        row["full_fixed_us"],
                        row["estimated_fixed_us"],
                        _fraction_text(row["difference_us"]),
                        _fraction_text(row["sampling_bound_us"]),
                        _fraction_text(row["quantization_bound_us"]),
                        _fraction_text(row["bucket_bound_us"]),
                        _fraction_text(row["total_error_bound_us"]),
                        "是" if row["estimated_value_in_error_bound"] else "否",
                    ),
                    tags=(tag,),
                )

        def _refresh_attribution(self) -> None:
            self.attribution_tree.delete(*self.attribution_tree.get_children())
            service = self._current_service_row()
            if not service:
                return
            rows = [
                row
                for row in service["attribution"]
                if row["policy_id"] == self.selected_policy
            ]
            bucket_counts = {
                row["bucket_index"]: row["record_count"]
                for row in service["buckets"]
                if row["policy_id"] == self.selected_policy
            }
            bucket_hits = {
                row["bucket_index"]: row["sample_hits"]
                for row in service["buckets"]
                if row["policy_id"] == self.selected_policy
            }
            for row in rows:
                if row["scope"] == "total":
                    label = "TOTAL"
                    count = sum(bucket_counts.values())
                    hits = sum(bucket_hits.values())
                else:
                    label = f"桶 {row['bucket_index']}"
                    count = bucket_counts.get(row["bucket_index"], "-")
                    hits = bucket_hits.get(row["bucket_index"], "-")
                contributions = row.get("contributions_us", {})
                self.attribution_tree.insert(
                    "",
                    "end",
                    values=(
                        label,
                        count,
                        hits,
                        _fraction_text(row.get("estimated_total_us")),
                        _fraction_text(row.get("full_total_us")),
                        _fraction_text(row.get("estimated_minus_true_us")),
                        _fraction_text(contributions.get("sampling_sparse")),
                        _fraction_text(contributions.get("fixed_truncation")),
                        _fraction_text(contributions.get("bucket_boundary")),
                        _fraction_text(row.get("total_error_bound_us")),
                        row.get("dominant_segment", "-"),
                    ),
                )
            self._refresh_flips()

        def _refresh_flips(self) -> None:
            self.flip_tree.delete(*self.flip_tree.get_children())
            self._flip_payloads.clear()
            if not self.view:
                return
            for index, flip in enumerate(self.view["flips"]):
                payload = flip["payload"]
                iid = f"flip-{index}"
                if flip["kind"] == "error_bound_boundary":
                    self.flip_tree.insert(
                        "",
                        "end",
                        iid=iid,
                        values=(
                            "误差界内判定",
                            f"{payload['left']} ↔ {payload['right']}",
                            payload["kind"],
                            _fraction_text(payload["absolute_difference_us"]),
                            _fraction_text(payload["combined_error_bound_us"]),
                            "受误差界影响",
                        ),
                    )
                else:
                    self.flip_tree.insert(
                        "",
                        "end",
                        iid=iid,
                        values=(
                            "参数敏感性翻转",
                            payload["policy_id"],
                            f"{payload['parameter']}: {payload['old_value']} → {payload['new_value']}",
                            "-",
                            _fraction_text(
                                payload["before"]["service_p95_error_bounds_us"].get(
                                    self.selected_service
                                )
                            ),
                            "排序/候选策略已翻转",
                        ),
                    )
                self._flip_payloads[iid] = payload

        def _on_flip_select(self, _event: Any = None) -> None:
            selection = self.flip_tree.selection()
            if not selection:
                return
            payload = self._flip_payloads.get(selection[0])
            self.flip_detail.delete("1.0", "end")
            self.flip_detail.insert(
                "1.0", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            )

        def _refresh_samples(self) -> None:
            if not self.selected_service or not self.services_doc:
                return
            service = self.services_doc.service_map()[self.selected_service]
            if self.selected_service not in self.records_by_service:
                self.records_by_service[self.selected_service] = generate_records(
                    service
                ).records
            records = self.records_by_service[self.selected_service]
            needle = self.sample_filter.get().strip().lower()
            rows = []
            for index, record in enumerate(records):
                if needle and needle not in record["client_key"].lower() and needle not in record["trace_id"].lower():
                    continue
                rows.append((index, record))
            sort_key = self.sample_sort.get()
            if sort_key == "duration_us":
                rows.sort(key=lambda item: (item[1]["duration_us"], item[0]))
            elif sort_key == "client_key":
                rows.sort(key=lambda item: (item[1]["client_key"], item[0]))
            total_pages = max(1, (len(rows) + self.sample_page_size - 1) // self.sample_page_size)
            self.sample_page = max(0, min(self.sample_page, total_pages - 1))
            start = self.sample_page * self.sample_page_size
            end = start + self.sample_page_size
            self.sample_tree.delete(*self.sample_tree.get_children())
            decisions = self._sample_decisions()
            for index, record in rows[start:end]:
                decision = decisions.get(index, "-")
                self.sample_tree.insert(
                    "",
                    "end",
                    values=(
                        index,
                        record["timestamp_ms"],
                        record["magnitude"],
                        record["client_key"],
                        record["duration_us"],
                        record["trace_id"],
                        "是" if record["sampled"] else "否",
                        decision,
                    ),
                )
            self.sample_page_label.configure(
                text=f"{self.sample_page + 1}/{total_pages} · {len(rows)} 条"
            )

        def _sample_decisions(self) -> dict[int, str]:
            service = self._current_service_row()
            if not service or not self.selected_policy:
                return {}
            summary = service["summary"]["policies"].get(self.selected_policy)
            if not summary:
                return {}
            policy = self._selected_policy_spec()
            records = self.records_by_service[self.selected_service]
            from .sampling import sample_records

            sampling = sample_records(records, policy, self.policies_doc.version)
            return {
                decision.record_index: decision.reason for decision in sampling.decisions
            }

        def _apply_sample_filter(self) -> None:
            self.sample_page = 0
            self._refresh_samples()

        def _page_samples(self, delta: int) -> None:
            self.sample_page += delta
            self._refresh_samples()

        def _refresh_view(self) -> None:
            try:
                self._load_from_ledger()
                self.logger.event("gui_view_refreshed", context={"run_id": self.view["run"]["run_id"] if self.view else None})
            except Exception as exc:
                self._handle_error(exc)

        def _analyze_current(self) -> None:
            if not self.services_doc or not self.policies_doc:
                self._load_documents_from_files()
            if not self.services_doc or not self.policies_doc:
                return
            try:
                result = analyze_and_persist(
                    self.services_doc,
                    self.policies_doc,
                    self.db_path,
                    logger=self.logger,
                )
                self._load_from_ledger()
                self.status_var.set(
                    f"已整体替换为 {result.run_id} · 参数翻转 {len(result.bundle.flip_events)} 条"
                )
            except Exception as exc:
                self._handle_error(exc)

        def _recover(self) -> None:
            try:
                result = recover_latest(self.db_path, logger=self.logger)
                self._load_from_ledger()
                self.status_var.set(
                    f"恢复完成 {result.run_id} · 新提交 {', '.join(result.newly_committed) or '无'} · "
                    f"复用已提交 {', '.join(result.skipped_committed) or '无'}"
                )
            except Exception as exc:
                self._handle_error(exc)

        def _verify(self) -> None:
            try:
                result = verify_ledger_run(self.db_path)
                self.logger.event("replay_verified", context=result)
                messagebox.showinfo(
                    "重放一致",
                    f"run {result['run_id']}\n"
                    f"服务 {result['services_checked']} 个\n"
                    f"逐桶比较 {result['bucket_checks']} 项\n"
                    f"snapshot {result['snapshot_digest']}",
                )
            except Exception as exc:
                self._handle_error(exc)

        def _apply_policy_edits(self) -> None:
            if not self.policies_doc or not self.selected_policy:
                return
            try:
                raw = self.policies_doc.to_dict()
                raw["version"] = self.policies_doc.version + 1
                for item in raw["policies"]:
                    if item["id"] != self.selected_policy:
                        continue
                    item["head_ratio_bp"] = int(self.policy_entries["head_ratio_bp"].get())
                    item["tail_duration_threshold_us"] = int(
                        self.policy_entries["tail_duration_threshold_us"].get()
                    )
                    item["tail_ratio_bp"] = int(self.policy_entries["tail_ratio_bp"].get())
                    item["scale_power"] = int(self.policy_entries["scale_power"].get())
                    item["bucket_count"] = int(self.policy_entries["bucket_count"].get())
                    item["description"] = self.policy_description.get()
                policies_doc = parse_policies_document(raw)
                write_json_atomic(self.policies_path, policies_doc.to_dict())
                self.policies_doc = policies_doc
                result = analyze_and_persist(
                    self.services_doc,
                    self.policies_doc,
                    self.db_path,
                    logger=self.logger,
                )
                self._load_from_ledger()
                self.status_var.set(
                    f"策略 v{policies_doc.version} 已提交为 {result.run_id}"
                )
            except Exception as exc:
                self._handle_error(exc)

        def _apply_service_description(self) -> None:
            if not self.services_doc or not self.selected_service:
                return
            try:
                updated = tuple(
                    replace(service, description=self.service_description.get())
                    if service.id == self.selected_service
                    else service
                    for service in self.services_doc.services
                )
                services_doc = ServicesDoc(
                    version=self.services_doc.version + 1,
                    services=updated,
                    description=self.services_doc.description,
                )
                write_json_atomic(self.services_path, services_doc.to_dict())
                self.services_doc = services_doc
                result = analyze_and_persist(
                    self.services_doc,
                    self.policies_doc,
                    self.db_path,
                    logger=self.logger,
                )
                self._load_from_ledger()
                self.status_var.set(
                    f"样本描述 v{services_doc.version} 已提交为 {result.run_id}"
                )
            except Exception as exc:
                self._handle_error(exc)

        def _handle_error(self, exc: Exception) -> None:
            if isinstance(exc, AppError):
                code = exc.code
                path = exc.path
                message = exc.message
                context = exc.context
            else:
                code = "E_ENV"
                path = "$"
                message = str(exc)
                context = {}
            self.logger.event(
                "gui_error",
                level="ERROR",
                code=code,
                path=path,
                message=message,
                context=context,
            )
            self.status_var.set(f"{code} · {path} · {message}")
            try:
                messagebox.showerror(code, f"{path}\n{message}")
            except Exception:
                pass

    app = TraceDiagApp()
    app.mainloop()
