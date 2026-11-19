from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from joinlab.errors import JoinLabError
from joinlab.logging_utils import StructuredLogger
from joinlab.params import QUANTILE_POLICIES, load_params_file, normalise_params
from joinlab.session import WorkbenchSession
from joinlab.spec import load_workspace


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "samples" / "base_tables.json"
DEFAULT_SESSION = ROOT / "samples" / "session.json"
DEFAULT_DB = ROOT / "artifacts" / "joinlab.sqlite"
DEFAULT_SNAPSHOT = ROOT / "artifacts" / "snapshot.json"
DEFAULT_STATE = ROOT / "artifacts" / "version_state.json"
DEFAULT_LOG = ROOT / "artifacts" / "logs" / "joinlab.jsonl"


class WorkbenchApp:
    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("JoinLab · 等值联接规模估计与采样诊断")
        self.root.geometry("1280x860")
        self.root.minsize(1040, 700)
        self.session: WorkbenchSession | None = None
        self.report: dict[str, Any] | None = None
        self.logger = StructuredLogger(DEFAULT_LOG)

        self.workspace_var = tk.StringVar(value=str(DEFAULT_WORKSPACE))
        self.session_file_var = tk.StringVar(value=str(DEFAULT_SESSION))
        self.db_var = tk.StringVar(value=str(DEFAULT_DB))
        self.snapshot_var = tk.StringVar(value=str(DEFAULT_SNAPSHOT))
        self.state_var = tk.StringVar(value=str(DEFAULT_STATE))
        self.bucket_var = tk.StringVar(value="10")
        self.sample_var = tk.StringVar(value="320")
        self.seed_var = tk.StringVar(value="20261120")
        self.policy_var = tk.StringVar(value="nearest_rank")
        self.k_var = tk.StringVar(value="5")
        self.status_var = tk.StringVar(value="就绪")

        self._build_ui()

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        paths = ttk.LabelFrame(outer, text="本地文件", padding=8)
        paths.pack(fill="x")
        row = 0
        for label, variable, button_text, command in (
            ("工作区 JSON", self.workspace_var, "选择", self._choose_workspace),
            ("会话参数", self.session_file_var, "选择", self._choose_session),
            ("SQLite", self.db_var, None, None),
            ("快照", self.snapshot_var, None, None),
            ("版本状态", self.state_var, None, None),
        ):
            ttk.Label(paths, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            ttk.Entry(paths, textvariable=variable, width=100).grid(row=row, column=1, sticky="ew", pady=2)
            if button_text:
                ttk.Button(paths, text=button_text, command=command).grid(row=row, column=2, padx=4, pady=2)
            row += 1
        paths.columnconfigure(1, weight=1)

        params = ttk.LabelFrame(outer, text="参数（变化即整体替换）", padding=8)
        params.pack(fill="x", pady=(8, 0))
        ttk.Label(params, text="分桶数").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.bucket_var, width=10).grid(row=0, column=1, padx=(4, 16))
        ttk.Label(params, text="抽样规模").grid(row=0, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.sample_var, width=12).grid(row=0, column=3, padx=(4, 16))
        ttk.Label(params, text="种子").grid(row=0, column=4, sticky="w")
        ttk.Entry(params, textvariable=self.seed_var, width=16).grid(row=0, column=5, padx=(4, 16))
        ttk.Label(params, text="分位口径").grid(row=0, column=6, sticky="w")
        ttk.Combobox(params, textvariable=self.policy_var, values=QUANTILE_POLICIES, state="readonly", width=16).grid(row=0, column=7, padx=(4, 16))
        ttk.Label(params, text="Top K").grid(row=0, column=8, sticky="w")
        ttk.Entry(params, textvariable=self.k_var, width=6).grid(row=0, column=9, padx=4)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="装载并全量重算", command=self.load_and_build).pack(side="left")
        ttk.Button(actions, text="应用参数并整体替换", command=self.apply_parameters).pack(side="left", padx=6)
        ttk.Button(actions, text="当前参数重算", command=self.recompute).pack(side="left", padx=6)
        ttk.Button(actions, text="保存快照", command=self.save_snapshot).pack(side="left", padx=6)
        ttk.Button(actions, text="恢复快照并逐项校验", command=self.restore_snapshot).pack(side="left", padx=6)
        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

        summary = ttk.LabelFrame(outer, text="统计摘要", padding=6)
        summary.pack(fill="x")
        self.summary_tree = ttk.Treeview(
            summary,
            columns=("rows", "buckets", "hist_digest", "sample_digest"),
            show="tree headings",
            height=3,
        )
        self.summary_tree.heading("#0", text="表")
        self.summary_tree.heading("rows", text="过滤后行数")
        self.summary_tree.heading("buckets", text="桶数")
        self.summary_tree.heading("hist_digest", text="直方图摘要")
        self.summary_tree.heading("sample_digest", text="样本摘要")
        self.summary_tree.column("#0", width=140)
        self.summary_tree.column("rows", width=100, anchor="e")
        self.summary_tree.column("buckets", width=70, anchor="e")
        self.summary_tree.column("hist_digest", width=320)
        self.summary_tree.column("sample_digest", width=320)
        self.summary_tree.pack(fill="x")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True, pady=(8, 0))
        self.text_widgets: dict[str, Any] = {}
        for name in ("估算", "归因", "真值", "版本与陈旧", "完整 JSON"):
            frame = ttk.Frame(notebook)
            text = tk.Text(frame, wrap="none", font=("Consolas", 10), undo=False)
            y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
            text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            text.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            notebook.add(frame, text=name)
            self.text_widgets[name] = text

    def _choose_workspace(self) -> None:
        from tkinter import filedialog

        value = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if value:
            self.workspace_var.set(value)

    def _choose_session(self) -> None:
        from tkinter import filedialog

        value = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if value:
            self.session_file_var.set(value)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _show_error(self, error: JoinLabError) -> None:
        self.logger.error(error.code, error.message, field_path=error.field_path, details=error.details)
        self._set_status(f"{error.code}: {error.message}")
        payload = error.to_dict()
        for name, widget in self.text_widgets.items():
            widget.delete("1.0", "end")
            if name in {"估算", "完整 JSON"}:
                widget.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))
        if self.report is not None:
            self._render_report(self.report)

    def _params_from_ui(self) -> Any:
        raw = {
            "bucket_count": int(self.bucket_var.get()),
            "sample_size": int(self.sample_var.get()),
            "seed": int(self.seed_var.get()),
            "quantile_policy": self.policy_var.get(),
            "attribution_k": int(self.k_var.get()),
        }
        return normalise_params(raw, None, 5)

    def load_and_build(self) -> None:
        try:
            spec = load_workspace(self.workspace_var.get())
            raw_params = dict(spec.default_params)
            raw_params.update(load_params_file(self.session_file_var.get()))
            params = normalise_params(raw_params, None, spec.attribution_k)
            self.session = WorkbenchSession(
                spec=spec,
                db_path=self.db_var.get(),
                params=params,
                logger=self.logger,
                version_state_path=self.state_var.get(),
            )
            self.bucket_var.set(str(params.bucket_count))
            self.sample_var.set(str(params.sample_size))
            self.seed_var.set(str(params.seed))
            self.policy_var.set(params.quantile_policy)
            self.k_var.set(str(params.attribution_k))
            self.report = self.session.build_current()
            self._render_report(self.report)
            self._set_status(f"已装载，参数版本 {self.report['params_version']}，校验 {self.report['validation']['status']}")
        except JoinLabError as error:
            self._show_error(error)
        except ValueError as error:
            self._set_status(str(error))

    def apply_parameters(self) -> None:
        if self.session is None:
            self._set_status("请先装载工作区")
            return
        try:
            params = self._params_from_ui()
            result = self.session.replace_parameters(params)
            self.report = result.get("report", self.session.current_report())
            self._render_report(self.report)
            changed = result.get("changed")
            self._set_status("参数未变化" if not changed else f"已整体替换到版本 {self.report['params_version']}")
        except JoinLabError as error:
            if self.session.current is not None:
                self.report = self.session.current_report()
            self._show_error(error)
        except ValueError as error:
            self._set_status(str(error))

    def recompute(self) -> None:
        if self.session is None:
            self._set_status("请先装载工作区")
            return
        try:
            self.report = self.session.build_current()
            self._render_report(self.report)
            self._set_status(f"已重算，校验 {self.report['validation']['status']}")
        except JoinLabError as error:
            if self.session.current is not None:
                self.report = self.session.current_report()
            self._show_error(error)

    def save_snapshot(self) -> None:
        if self.session is None:
            self._set_status("请先装载工作区")
            return
        try:
            saved = self.session.save_snapshot(self.snapshot_var.get())
            self._set_status(f"快照已保存：版本 {saved['params_version']}")
        except JoinLabError as error:
            self._show_error(error)

    def restore_snapshot(self) -> None:
        if self.session is None:
            self._set_status("请先装载工作区")
            return
        try:
            restored = self.session.restore_snapshot(self.snapshot_var.get())
            self.report = restored["report"]
            params = self.report["params"]
            self.bucket_var.set(str(params["bucket_count"]))
            self.sample_var.set(str(params["sample_size"]))
            self.seed_var.set(str(params["seed"]))
            self.policy_var.set(params["quantile_policy"])
            self.k_var.set(str(params["attribution_k"]))
            self._render_report(self.report)
            self._set_status("快照已整体恢复，逐项重算校验通过")
        except JoinLabError as error:
            self._show_error(error)

    def _render_report(self, report: dict[str, Any]) -> None:
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        for table_name in (report["join"]["left_table"], report["join"]["right_table"]):
            table = report["tables"][table_name]
            histogram = report["histograms"][table_name]
            sample = report["samples"][table_name]
            self.summary_tree.insert(
                "",
                "end",
                text=table_name,
                values=(
                    table["filtered_row_count"],
                    histogram["bucket_count"],
                    histogram["digest"][:24] + "…",
                    sample["sample_digest"][:24] + "…",
                ),
            )
        estimate_view = {
            "params_version": report["params_version"],
            "params": report["params"],
            "histogram_path": report["estimate"]["histogram_path"],
            "sample_path": report["estimate"]["sample_path"],
            "difference_fixed": report["estimate"]["difference_fixed"],
            "difference_rows": report["estimate"]["difference_rows"],
            "truth_rows": report["truth"]["true_row_count"],
            "validation": report["validation"],
        }
        attribution_view = {
            "params_version": report["params_version"],
            "attribution": report["attribution"],
            "note": "排序：|contribution diff| DESC, key lower bound ASC, bucket ids ASC",
        }
        truth_view = {
            "true_row_count": report["truth"]["true_row_count"],
            "grouped_row_count": report["truth"]["grouped_row_count"],
            "output_sequence_rule": report["truth"]["output_sequence_rule"],
            "output_sequence_digest": report["truth"]["output_sequence_digest"],
            "grouped_distribution_digest": report["truth"]["grouped_distribution_digest"],
            "output_preview": report["truth"]["output_preview"],
            "grouped_distribution_head": report["truth"]["grouped_distribution"][:100],
        }
        history_view = {
            "replacement": report.get("replacement"),
            "stale_epochs": report.get("stale_epochs", []),
            "guard": "stale_epochs 仅审计；当前 histograms/samples/estimate/attribution/validation 全部来自同一参数版本",
        }
        outputs = {
            "估算": estimate_view,
            "归因": attribution_view,
            "真值": truth_view,
            "版本与陈旧": history_view,
            "完整 JSON": report,
        }
        for name, payload in outputs.items():
            widget = self.text_widgets[name]
            widget.delete("1.0", "end")
            widget.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    try:
        import tkinter as tk
    except ModuleNotFoundError:
        print("tkinter 不可用。Windows 官方 Python 安装时请勾选 Tcl/Tk；也可使用 python demo.py run。")
        return 2
    root = tk.Tk()
    WorkbenchApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

