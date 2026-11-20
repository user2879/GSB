"""Tkinter desktop workbench for Windows-local diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .service import WorkbenchService
from .util import format_fixed
from .validation import load_master_records, load_replay_config


ROOT = Path(__file__).resolve().parents[1]


def _ns_to_ms_text(value_ns: int) -> str:
    sign = "-" if value_ns < 0 else ""
    whole, fraction = divmod(abs(value_ns), 1_000_000)
    return f"{sign}{whole}.{fraction // 1000:03d}"


class WorkbenchWindow:
    """Builds a deterministic, fixed-order desktop layout."""

    def __init__(self, tk: Any, ttk: Any, messagebox: Any, workspace: Path) -> None:
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.workspace = workspace
        self.fixture_dir = workspace / "samples"
        if not self.fixture_dir.exists():
            self.fixture_dir = ROOT / "samples"
        self.service = WorkbenchService(workspace)
        self.root = tk.Tk()
        self.root.title("主从复制延迟诊断工作台")
        self.root.geometry("1280x840")
        self.root.minsize(1040, 700)
        self.snapshot_paths: dict[str, str] = {}
        self._configure_style()
        self._build_widgets()
        self.refresh()

    def _configure_style(self) -> None:
        style = self.ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=23)
        style.configure("Mono.TLabel", font=("Consolas", 9))

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = self.ttk.Frame(self.root, padding=(14, 10, 14, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.ttk.Label(header, text="主从复制延迟诊断工作台", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.header_var = self.tk.StringVar(value="正在加载…")
        self.ttk.Label(header, textvariable=self.header_var, style="Mono.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )

        toolbar = self.ttk.Frame(self.root, padding=(14, 4, 14, 8))
        toolbar.grid(row=1, column=0, sticky="ew")
        actions = (
            ("刷新", self.refresh),
            ("首次回放", self.run_initial),
            ("参数变更重算", self.run_changed),
            ("保存快照", self.save_snapshot),
            ("恢复选中快照", self.restore_selected),
            ("历史重放校验", self.verify_selected),
        )
        for index, (label, command) in enumerate(actions):
            self.ttk.Button(toolbar, text=label, command=command).grid(
                row=0, column=index, padx=(0, 8), sticky="w"
            )

        notebook = self.ttk.Notebook(self.root)
        notebook.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 8))
        self.notebook = notebook
        self._build_log_tab(notebook)
        self._build_sample_tab(notebook)
        self._build_estimate_tab(notebook)
        self._build_snapshot_tab(notebook)

        self.status_var = self.tk.StringVar(value="")
        status = self.ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(8, 4)
        )
        status.grid(row=3, column=0, sticky="ew")

    def _tree(
        self,
        parent: Any,
        columns: tuple[tuple[str, str, int], ...],
        *,
        height: int = 12,
    ) -> tuple[Any, Any]:
        frame = self.ttk.Frame(parent)
        tree = self.ttk.Treeview(
            frame,
            columns=tuple(column[0] for column in columns),
            show="headings",
            height=height,
        )
        for name, title, width in columns:
            tree.heading(name, text=title)
            tree.column(name, width=width, minwidth=60, stretch=True, anchor="w")
        vertical = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return frame, tree

    def _build_log_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="1 日志与进度")
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)

        self.log_range_var = self.tk.StringVar(value="")
        self.ttk.Label(tab, textvariable=self.log_range_var, style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.ttk.Label(tab, text="回放运行与原子进度", style="Section.TLabel").grid(
            row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6)
        )
        log_frame, self.log_tree = self._tree(
            tab,
            (
                ("seq", "提交序号", 90),
                ("txn", "事务标识", 160),
                ("commit_ms", "提交毫秒", 120),
                ("bytes", "事务字节", 110),
            ),
            height=20,
        )
        log_frame.grid(row=1, column=0, sticky="nsew")
        progress_frame, self.progress_tree = self._tree(
            tab,
            (
                ("param", "参数版本", 135),
                ("cursor", "已回放序号", 105),
                ("batches", "完成批次", 95),
                ("status", "状态", 155),
                ("updated", "更新时间 UTC", 200),
            ),
            height=20,
        )
        progress_frame.grid(row=1, column=1, sticky="nsew", padx=(12, 0))

    def _build_sample_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="2 延迟样本分布")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        self.distribution_var = self.tk.StringVar(value="")
        self.ttk.Label(tab, textvariable=self.distribution_var, style="Mono.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.ttk.Label(
            tab,
            text="逐条真值样本（按提交序号固定升序）",
            style="Section.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))
        frame, self.sample_tree = self._tree(
            tab,
            (
                ("seq", "提交序号", 80),
                ("completed", "回放完成毫秒", 135),
                ("completed_ns", "完成单调纳秒", 170),
                ("delay", "延迟毫秒", 120),
                ("fixed", "定点值", 120),
                ("batch", "批次键", 80),
            ),
            height=20,
        )
        frame.grid(row=2, column=0, sticky="nsew")

    def _build_estimate_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="3 定点统计与误差")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        tab.rowconfigure(3, weight=2)
        self.estimate_var = self.tk.StringVar(value="")
        self.ttk.Label(tab, textvariable=self.estimate_var, style="Mono.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        stat_frame, self.stat_tree = self._tree(
            tab,
            (
                ("window", "窗口", 65),
                ("range", "序号区间", 110),
                ("count", "样本/总量", 100),
                ("decision", "判定", 110),
                ("mean", "均值定点", 115),
                ("p95", "P95 定点", 115),
                ("max", "最大值定点", 115),
                ("dominant", "主导段(均值/P95/最大值)", 260),
            ),
            height=8,
        )
        stat_frame.grid(row=1, column=0, sticky="nsew")
        self.ttk.Label(tab, text="分段误差上界（定点单位）", style="Section.TLabel").grid(
            row=2, column=0, sticky="w", pady=(10, 6)
        )
        error_frame, self.error_tree = self._tree(
            tab,
            (
                ("window", "窗口", 65),
                ("stat", "统计量", 75),
                ("sampling", "采样截断", 120),
                ("fixed", "定点截断", 120),
                ("rate", "回放速率离散化", 145),
                ("total", "总上界", 120),
                ("dominant", "主导段", 190),
            ),
            height=12,
        )
        error_frame.grid(row=3, column=0, sticky="nsew")

    def _build_snapshot_tab(self, notebook: Any) -> None:
        tab = self.ttk.Frame(notebook, padding=10)
        notebook.add(tab, text="4 快照与重放")
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=3)
        tab.rowconfigure(1, weight=1)
        self.ttk.Label(tab, text="不可变快照列表（按文件名稳定排序）", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.ttk.Label(tab, text="整体替换 / 历史重放报告", style="Section.TLabel").grid(
            row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 6)
        )
        frame, self.snapshot_tree = self._tree(
            tab,
            (
                ("file", "文件名", 255),
                ("created", "创建时间 UTC", 190),
                ("param", "参数版本", 135),
                ("samples", "样本", 70),
                ("ro", "只读", 60),
            ),
            height=18,
        )
        frame.grid(row=1, column=0, sticky="nsew")
        self.snapshot_tree.bind("<<TreeviewSelect>>", self._on_snapshot_select)
        report_frame = self.ttk.Frame(tab)
        report_frame.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        report_frame.rowconfigure(0, weight=1)
        report_frame.columnconfigure(0, weight=1)
        self.report_text = self.tk.Text(report_frame, wrap="word", font=("Consolas", 9))
        self.report_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = self.ttk.Scrollbar(report_frame, orient="vertical", command=self.report_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.report_text.configure(yscrollcommand=scrollbar.set, state="disabled")

    def _clear_tree(self, tree: Any) -> None:
        children = tree.get_children()
        if children:
            tree.delete(*children)

    def refresh(self) -> None:
        try:
            state = self.service.state_summary()
        except WorkbenchError as exc:
            self._show_error(exc)
            return
        log = state["log"]
        self.log_range_var.set(
            f"主库日志：{log['count']} 条，序号 {log['first_seq']}–{log['last_seq']}，"
            f"摘要 {str(log['digest'])[:16]}…"
        )
        self.header_var.set(
            f"SQLite: {self.service.db_path}    |    运行 {len(state['runs'])}    "
            f"|    最新参数 {state['latest_param_version']}"
        )

        self._clear_tree(self.log_tree)
        for record in self.service.repo.get_master_log():
            self.log_tree.insert(
                "",
                "end",
                values=(record.commit_seq, record.txn_id, record.commit_ts_ms, record.txn_bytes),
            )

        self._clear_tree(self.progress_tree)
        for item in sorted(state["progress"], key=lambda value: (value["updated_utc"], value["param_version"])):
            self.progress_tree.insert(
                "",
                "end",
                values=(
                    item["param_version"],
                    item["last_replayed_seq"],
                    item["completed_batches"],
                    item["status"],
                    item["updated_utc"],
                ),
            )

        self._render_samples(state)
        self._render_estimation(state)
        self._render_snapshots(state)
        recovery = state.get("recovery") or {}
        self.status_var.set(
            "就绪"
            if not recovery.get("requires_rerun")
            else "检测到未完整批次；已标记 UNKNOWN / REQUIRES_RERUN，需重跑"
        )

    def _render_samples(self, state: dict[str, Any]) -> None:
        self._clear_tree(self.sample_tree)
        param_version = state.get("latest_param_version")
        if param_version is None:
            self.distribution_var.set("尚无可显示的回放样本")
            return
        samples = self.service.repo.get_samples(param_version)
        scale_power = 3
        latest = state.get("latest_estimation")
        if latest:
            scale_power = latest["estimation"]["scale_power"]
        for sample in samples:
            self.sample_tree.insert(
                "",
                "end",
                values=(
                    sample.commit_seq,
                    _ns_to_ms_text(sample.replay_completed_ns),
                    sample.replay_completed_ns,
                    _ns_to_ms_text(sample.delay_ns),
                    format_fixed(sample.fixed_delay, scale_power),
                    sample.batch_id,
                ),
            )
        if not samples:
            self.distribution_var.set(f"参数 {param_version} 尚无样本")
            return
        values = [sample.delay_ns for sample in samples]
        minimum, maximum = min(values), max(values)
        bins = 5
        span = max(1, maximum - minimum + 1)
        width = (span + bins - 1) // bins
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, (value - minimum) // width)
            counts[index] += 1
        peak = max(counts)
        lines = [
            f"参数 {param_version}：{len(values)} 个样本，延迟 {_ns_to_ms_text(minimum)}–{_ns_to_ms_text(maximum)} ms"
        ]
        for index, count in enumerate(counts):
            start = minimum + index * width
            end = minimum + (index + 1) * width - 1
            bar = "#" * max(0, round(count * 28 / peak)) if peak else ""
            lines.append(
                f"[{_ns_to_ms_text(start):>10}, {_ns_to_ms_text(end):>10}] {bar:<28} {count}"
            )
        self.distribution_var.set("\n".join(lines))

    def _render_estimation(self, state: dict[str, Any]) -> None:
        self._clear_tree(self.stat_tree)
        self._clear_tree(self.error_tree)
        latest = state.get("latest_estimation")
        if not latest:
            self.estimate_var.set("尚无可显示的估算结果")
            return
        estimation = latest["estimation"]
        self.estimate_var.set(
            f"估算口径 {estimation['estimator_version']} | 缩放 10^{estimation['scale_power']} "
            f"| 截断 floor | P95 {estimation['p95_method']} | "
            f"指纹 {estimation['result_fingerprint'][:20]}…"
        )
        for window in sorted(estimation["windows"], key=lambda item: item["window_index"]):
            statistics = window["statistics"]
            if window["determinable"]:
                mean = statistics["mean"]["ms"]
                p95 = statistics["p95"]["ms"]
                maximum = statistics["max"]["ms"]
                budgets = window["error_budget"]
                dominant = " / ".join(
                    budgets[name]["dominant_segment"] for name in ("mean", "p95", "max")
                )
                decision = "可判定"
            else:
                mean = p95 = maximum = "不可判定"
                dominant = ",".join(window["decision_reasons"])
                decision = "不可判定"
            self.stat_tree.insert(
                "",
                "end",
                values=(
                    window["window_index"],
                    f"{window['first_commit_seq']}–{window['last_commit_seq']}",
                    f"{window['basis']['sample_count_used']}/{window['basis']['expected_count']}",
                    decision,
                    mean,
                    p95,
                    maximum,
                    dominant,
                ),
            )
            if not window["determinable"]:
                continue
            for stat_name in ("mean", "p95", "max"):
                budget = window["error_budget"][stat_name]
                segments = budget["segments"]
                self.error_tree.insert(
                    "",
                    "end",
                    values=(
                        window["window_index"],
                        stat_name,
                        segments["sampling_truncation"]["upper_bound_fixed"],
                        segments["fixed_point_truncation"]["upper_bound_fixed"],
                        segments["replay_rate_discretization"]["upper_bound_fixed"],
                        budget["total_upper_bound_fixed"],
                        budget["dominant_segment"],
                    ),
                )

    def _render_snapshots(self, state: dict[str, Any]) -> None:
        self._clear_tree(self.snapshot_tree)
        self.snapshot_paths.clear()
        for index, snapshot in enumerate(sorted(state["snapshots"], key=lambda item: item["filename"])):
            item_id = f"snapshot-{index}"
            self.snapshot_paths[item_id] = snapshot["path"]
            self.snapshot_tree.insert(
                "",
                "end",
                iid=item_id,
                values=(
                    snapshot["filename"],
                    snapshot.get("created_at_utc", "读取失败"),
                    snapshot.get("param_version", "—"),
                    snapshot.get("sample_count", "—"),
                    "是" if snapshot["readonly"] else "否",
                ),
            )

    def _selected_snapshot(self, *, quiet: bool = False) -> Path | None:
        selection = self.snapshot_tree.selection()
        if not selection:
            if not quiet:
                self.messagebox.showinfo("快照", "请先选择一个快照。")
            return None
        path = self.snapshot_paths.get(selection[0])
        return None if path is None else Path(path)

    def _set_report(self, value: Any) -> None:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", text)
        self.report_text.configure(state="disabled")

    def _on_snapshot_select(self, _event: Any) -> None:
        path = self._selected_snapshot(quiet=True)
        if path is None:
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            self._set_report(
                {
                    "snapshot_path": str(path),
                    "snapshot_format_version": document.get("snapshot_format_version"),
                    "log": document.get("log"),
                    "progress": document.get("progress"),
                    "estimation_fingerprint": (document.get("estimation") or {}).get(
                        "result_fingerprint"
                    ),
                    "note": "只读快照；恢复采用整体替换，不做增量合并。",
                }
            )
        except Exception as exc:  # noqa: BLE001
            self._set_report({"error": str(exc), "snapshot_path": str(path)})

    def run_initial(self) -> None:
        try:
            records = load_master_records(self.fixture_dir / "writes.json")
            params, spec, _ = load_replay_config(self.fixture_dir / "replay_params.json")
            self.service.import_master_log(records, source=str(self.fixture_dir / "writes.json"))
            self.service.run_replay(params, spec)
            self.service.estimate(params.version)
            self.refresh()
            self.notebook.select(2)
        except WorkbenchError as exc:
            self._show_error(exc)

    def run_changed(self) -> None:
        try:
            records = load_master_records(self.fixture_dir / "writes.json")
            if self.service.repo.log_stats()["count"] == 0:
                self.service.import_master_log(records, source=str(self.fixture_dir / "writes.json"))
            params, spec, _ = load_replay_config(self.fixture_dir / "replay_params_changed.json")
            self.service.run_replay(params, spec)
            self.service.estimate(params.version)
            self.refresh()
            self.notebook.select(2)
        except WorkbenchError as exc:
            self._show_error(exc)

    def save_snapshot(self) -> None:
        try:
            report = self.service.save_snapshot()
            self.refresh()
            self._set_report(report)
            self.notebook.select(3)
        except WorkbenchError as exc:
            self._show_error(exc)

    def restore_selected(self) -> None:
        path = self._selected_snapshot()
        if path is None:
            return
        try:
            report = self.service.restore_snapshot(path)
            self.refresh()
            self._set_report(report)
            self.notebook.select(3)
        except WorkbenchError as exc:
            self._show_error(exc)

    def verify_selected(self) -> None:
        path = self._selected_snapshot()
        if path is None:
            return
        report = self.service.verify_snapshot(path)
        self._set_report(report)
        self.notebook.select(3)
        if not report.get("reproducible"):
            self.messagebox.showerror(report.get("code", "E_REPLAY"), json.dumps(report, ensure_ascii=False, indent=2))

    def _show_error(self, exc: WorkbenchError) -> None:
        self.service.logger.event("ui_business_error", **exc.as_dict())
        self.messagebox.showerror(exc.code, f"{exc.message}\n出错序号: {exc.seq}\n字段: {exc.field}")
        self.status_var.set(f"{exc.code}: {exc.message}")

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.service.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args(argv)

    tk_spec = importlib.util.find_spec("tkinter")
    numpy_spec = importlib.util.find_spec("numpy")
    if args.check_imports:
        print(
            json.dumps(
                {
                    "tkinter_available": tk_spec is not None,
                    "numpy_available": numpy_spec is not None,
                    "python": sys.version,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0 if tk_spec is not None else 1
    if tk_spec is None:
        print(
            "E_ENV: tkinter 不可用。Windows 官方 Python 安装时请勾选 tcl/tk 组件。",
            file=sys.stderr,
        )
        return 2
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # noqa: BLE001
        print(f"E_ENV: tkinter 导入失败: {exc}", file=sys.stderr)
        return 2
    window: WorkbenchWindow | None = None
    try:
        window = WorkbenchWindow(tk, ttk, messagebox, Path(args.workspace).resolve())
        window.run()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"E_UI: 无法启动 Tkinter 工作台: {exc}", file=sys.stderr)
        return 3
    finally:
        if window is not None:
            window.close()


if __name__ == "__main__":
    raise SystemExit(main())
