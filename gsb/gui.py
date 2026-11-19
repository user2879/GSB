from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .errors import GSBError
from .logging_store import StructuredLogger
from .session import Session, restore_snapshot_into


class JoinLabApp:
    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.session: Session | None = None
        project_root = Path(__file__).resolve().parents[1]
        self.logger = StructuredLogger(
            project_root / "var" / "gui_logs" / "events.sqlite3",
            project_root / "var" / "gui_logs" / "events.jsonl",
        )
        self.workdir = project_root / "var" / "gui_work"
        self.spec_var = tk.StringVar(value=str(project_root / "examples" / "demo_skewed.json"))
        self.buckets_var = tk.StringVar(value="16")
        self.sample_size_var = tk.StringVar(value="500")
        self.sample_seed_var = tk.StringVar(value="424242")
        self.top_k_var = tk.StringVar(value="8")
        self.int_rule_var = tk.StringVar(value="left_closed_right_open")
        self.float_rule_var = tk.StringVar(value="left_open_right_closed")
        self.version_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="等待加载场景")
        self._build()
        self.root.after(200, self._auto_load)

    def _build(self) -> None:
        ttk = self.ttk
        self.root.title("GSB 等值联接规模估算与采样诊断实验台")
        self.root.geometry("1280x820")
        style = ttk.Style()
        style.configure("Treeview", rowheight=24)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Status.TLabel", background="#eef3f8", padding=6)

        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="等值联接规模估算与采样诊断实验台", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.version_var).pack(side="right")

        load_row = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        load_row.pack(fill="x")
        ttk.Label(load_row, text="场景 JSON").pack(side="left")
        ttk.Entry(load_row, textvariable=self.spec_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(load_row, text="浏览", command=self._browse_spec).pack(side="left", padx=3)
        ttk.Button(load_row, text="加载并运行", command=self.load_and_run).pack(side="left", padx=3)
        ttk.Button(load_row, text="重新运行", command=self.run_current).pack(side="left", padx=3)

        params = ttk.LabelFrame(self.root, text="会话参数", padding=8)
        params.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(params, text="分桶数").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(params, from_=1, to=4096, textvariable=self.buckets_var, width=10).grid(
            row=0, column=1, padx=(4, 16)
        )
        ttk.Label(params, text="抽样规模").grid(row=0, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.sample_size_var, width=12).grid(row=0, column=3, padx=(4, 16))
        ttk.Label(params, text="抽样种子").grid(row=0, column=4, sticky="w")
        ttk.Entry(params, textvariable=self.sample_seed_var, width=14).grid(row=0, column=5, padx=(4, 16))
        ttk.Label(params, text="Top K").grid(row=0, column=6, sticky="w")
        ttk.Spinbox(params, from_=1, to=100, textvariable=self.top_k_var, width=8).grid(row=0, column=7, padx=4)
        ttk.Label(params, text="整数键边界").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            params,
            textvariable=self.int_rule_var,
            values=["left_closed_right_open", "left_open_right_closed"],
            state="readonly",
            width=25,
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(params, text="浮点键边界").grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Combobox(
            params,
            textvariable=self.float_rule_var,
            values=["left_closed_right_open", "left_open_right_closed"],
            state="readonly",
            width=25,
        ).grid(row=1, column=4, columnspan=2, sticky="w", padx=4, pady=(8, 0))
        ttk.Button(params, text="应用参数并整体替换", command=self.apply_parameters).grid(
            row=1, column=6, columnspan=2, padx=4, pady=(8, 0)
        )

        actions = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        actions.pack(fill="x")
        ttk.Button(actions, text="保存快照", command=self.save_snapshot).pack(side="left")
        ttk.Button(actions, text="恢复快照并重放", command=self.restore_snapshot).pack(side="left", padx=6)
        ttk.Button(actions, text="刷新日志", command=self.refresh_logs).pack(side="left", padx=6)
        ttk.Label(actions, text="所有数据与快照仅保存在本地文件系统").pack(side="right")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.overview_text = self._text_tab("总览")
        self.hist_tree = self._tree_tab(
            "直方图桶",
            ("side", "index", "interval", "rows", "nulls", "digest"),
            ("侧", "#", "区间", "行数", "空值", "摘要"),
        )
        self.attr_tree = self._tree_tab(
            "分桶归因",
            ("pair", "left", "right", "hist", "sample", "diff", "source", "basis"),
            ("桶对", "左下界", "右下界", "直方图", "抽样", "差异", "来源", "依据"),
        )
        self.truth_text = self._text_tab("真实执行")
        self.replay_text = self._text_tab("快照 / 日志")

        status = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", anchor="w")
        status.pack(fill="x", side="bottom")

    def _text_tab(self, title: str) -> Any:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        text = self.tk.Text(frame, wrap="none", font=("Consolas", 10))
        y_scroll = self.ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        x_scroll = self.ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return text

    def _tree_tab(self, title: str, columns: tuple[str, ...], headings: tuple[str, ...]) -> Any:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings")
        for column, heading in zip(columns, headings):
            tree.heading(column, text=heading)
            width = 90
            if column in {"basis"}:
                width = 360
            elif column in {"interval", "source", "digest"}:
                width = 150
            tree.column(column, width=width, anchor="w")
        y_scroll = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _browse_spec(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(title="选择场景 JSON", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if selected:
            self.spec_var.set(selected)
            self.load_and_run()

    def _auto_load(self) -> None:
        path = Path(self.spec_var.get())
        if path.exists():
            try:
                self.load_and_run()
            except Exception:
                pass
        else:
            self.status_var.set("请选择本地场景 JSON")

    def _set_session(self, session: Session) -> None:
        self.session = session
        self.workdir = session.workdir
        join = session.spec.join
        self.buckets_var.set(str(join.buckets))
        self.sample_size_var.set(str(join.sample_size))
        self.sample_seed_var.set(str(join.sample_seed))
        self.top_k_var.set(str(join.top_k))
        self.int_rule_var.set(join.boundary_rules["int"])
        self.float_rule_var.set(join.boundary_rules["float"])
        self.version_var.set(f"参数版本 v{join.parameter_version}")
        self.render()

    def load_and_run(self) -> None:
        try:
            spec_path = Path(self.spec_var.get())
            session = Session.from_spec_file(spec_path, self.workdir, logger=self.logger)
            session.run()
            self._set_session(session)
            self.status_var.set(f"已加载 {spec_path.name}；参数版本 v{session.parameter_version}")
        except Exception as exc:
            self._show_error(exc)

    def run_current(self) -> None:
        if self.session is None:
            self.load_and_run()
            return
        try:
            self.session.run()
            self._set_session(self.session)
            self.status_var.set(f"已重新运行；参数版本 v{self.session.parameter_version}")
        except Exception as exc:
            self._show_error(exc)

    def _parse_int(self, value: str, field: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise GSBError("E_SPEC", f"{field} 必须为整数", {"value": value}) from exc

    def apply_parameters(self) -> None:
        if self.session is None:
            self.load_and_run()
            return
        try:
            evolution = self.session.evolve(
                buckets=self._parse_int(self.buckets_var.get(), "分桶数"),
                sample_size=self._parse_int(self.sample_size_var.get(), "抽样规模"),
                sample_seed=self._parse_int(self.sample_seed_var.get(), "抽样种子"),
                boundary_rules={
                    "int": self.int_rule_var.get(),
                    "float": self.float_rule_var.get(),
                },
                top_k=self._parse_int(self.top_k_var.get(), "Top K"),
            )
            self._set_session(self.session)
            transition = evolution.transition
            if transition.get("no_op"):
                self.status_var.set("参数未变化，当前视图保持不变")
            else:
                self.status_var.set(
                    "参数版本 v{} → v{}，整体替换完成；受影响桶对 {} 个".format(
                        transition["old_parameter_version"],
                        transition["new_parameter_version"],
                        transition["affected_pair_count"],
                    )
                )
        except Exception as exc:
            self._show_error(exc)

    def save_snapshot(self) -> None:
        if self.session is None:
            self.status_var.set("尚无会话可保存")
            return
        from tkinter import filedialog

        selected = filedialog.asksaveasfilename(
            title="保存会话快照",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="join_session_snapshot.json",
        )
        if not selected:
            return
        try:
            meta = self.session.save_snapshot(selected)
            self.status_var.set(f"快照已保存：{Path(selected).name}")
            self.replay_text.delete("1.0", "end")
            self.replay_text.insert("end", json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as exc:
            self._show_error(exc)

    def restore_snapshot(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(title="选择会话快照", filetypes=[("JSON", "*.json")])
        if not selected:
            return
        try:
            session, transition = restore_snapshot_into(
                selected,
                self.workdir,
                current=self.session,
                logger=self.logger,
            )
            self._set_session(session)
            self.status_var.set(
                "快照恢复与重放通过；参数版本 v{}；整体替换={}".format(
                    session.parameter_version,
                    transition.get("version_changed", True),
                )
            )
            self.replay_text.delete("1.0", "end")
            self.replay_text.insert("end", json.dumps(transition, ensure_ascii=False, indent=2))
        except Exception as exc:
            self._show_error(exc)

    def refresh_logs(self) -> None:
        events = self.logger.read_events(100)
        self.replay_text.delete("1.0", "end")
        self.replay_text.insert("end", json.dumps(events, ensure_ascii=False, indent=2))

    def render(self) -> None:
        if self.session is None or self.session.current is None:
            return
        result = self.session.current
        report = self.session.current_report()
        decision = result.decision
        estimates = decision["estimates"]
        interval = decision["interval"]
        overview = [
            f"场景: {report['scenario']}",
            f"参数版本: v{report['parameters']['parameter_version']}",
            f"规范摘要: {report['spec_digest']}",
            f"数据摘要: {report['data']['scenario_data_digest']}",
            "",
            f"直方图路径: {estimates['histogram']:.6f}  (fixed={result.histogram_estimate['estimate_fixed']})",
            f"抽样路径:   {estimates['sampling']:.6f}  (fixed={result.sampling_estimate['estimate_fixed']})",
            f"真实执行:   {estimates['truth']}",
            f"容差区间:   [{interval['lower']}, {interval['upper']}]  rel={interval['relative_tolerance']:.6f}",
            f"判定:       {decision['outcome']}",
            f"选择路径:   {decision['selected_path']}",
            f"选择依据:   {decision['path_selection']['reason']}",
            "",
            f"重叠桶对: {result.histogram_estimate['overlap_pair_count']}",
            f"样本命中对: {result.sampling_estimate['sample_hit_pairs']}",
            f"样本共同键: {result.sampling_estimate['observed_common_keys']}",
            f"左表重标定系数: {result.sampling_estimate['left_coefficient']:.6f}",
            f"右表重标定系数: {result.sampling_estimate['right_coefficient']:.6f}",
            "",
            f"直方图越界比例: {decision['histogram']['out_of_bounds_ratio']:.6f}",
            f"抽样越界比例:   {decision['sampling']['out_of_bounds_ratio']:.6f}",
            f"分析摘要: {result.analysis_digest}",
        ]
        if decision.get("dominant_deviation_bucket"):
            overview.append(
                "主导偏差桶: {}".format(
                    json.dumps(decision["dominant_deviation_bucket"], ensure_ascii=False)
                )
            )
        if result.report.get("last_transition"):
            overview.append("")
            overview.append("最近参数演进:")
            overview.append(json.dumps(result.report["last_transition"], ensure_ascii=False, indent=2))
        self.overview_text.delete("1.0", "end")
        self.overview_text.insert("end", "\n".join(overview))

        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)
        for side in ("left", "right"):
            histogram = result.histograms[side]
            for bucket in histogram.buckets:
                self.hist_tree.insert(
                    "",
                    "end",
                    values=(
                        side,
                        bucket.index,
                        bucket.interval_label(),
                        bucket.row_count,
                        bucket.null_count,
                        bucket.digest[:16],
                    ),
                )

        for item in self.attr_tree.get_children():
            self.attr_tree.delete(item)
        for record in result.attribution["top_k"]:
            self.attr_tree.insert(
                "",
                "end",
                values=(
                    record["pair_id"],
                    record["left_lower"],
                    record["right_lower"],
                    f"{record['histogram_estimate']:.3f}",
                    f"{record['sampling_estimate']:.3f}",
                    f"{record['estimate_difference']:.3f}",
                    record["source"],
                    record["basis"],
                ),
            )

        truth = result.truth
        truth_lines = [
            f"真实输出行数: {truth['row_count']}",
            f"共同键数: {truth['common_key_count']}",
            f"左/右过滤后非空键行数: {truth['left_filtered_rows']} / {truth['right_filtered_rows']}",
            f"真值摘要: {truth['digest']}",
            "",
            "按真实贡献前 50 键:",
            json.dumps(truth["key_distribution_top"], ensure_ascii=False, indent=2),
            "",
            "左桶真实行数:",
            json.dumps(truth["left_bucket_rows"], ensure_ascii=False),
            "右桶真实行数:",
            json.dumps(truth["right_bucket_rows"], ensure_ascii=False),
        ]
        self.truth_text.delete("1.0", "end")
        self.truth_text.insert("end", "\n".join(truth_lines))

    def _show_error(self, error: Exception) -> None:
        from tkinter import messagebox

        if isinstance(error, GSBError):
            self.logger.error(error, "gui_error")
            message = f"{error.code}: {error.message}\n\n{json.dumps(error.details, ensure_ascii=False, indent=2)}"
            self.status_var.set(f"{error.code}: {error.message}")
        else:
            message = f"{type(error).__name__}: {error}"
            self.status_var.set(message)
        messagebox.showerror("操作失败", message)


def main() -> int:
    try:
        import tkinter as tk
    except ImportError:
        print(
            "当前 Python 未包含 tkinter。Windows 官方 Python 安装包请勾选 Tcl/Tk；"
            "也可使用 `python -m gsb run --spec examples/demo_skewed.json` 运行 CLI。",
            file=sys.stderr,
        )
        return 2
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"无法创建 Tk 窗口: {exc}", file=sys.stderr)
        return 2
    JoinLabApp(root)
    root.mainloop()
    return 0
