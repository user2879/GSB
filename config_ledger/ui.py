"""Tkinter diagnostics console (imported lazily so CLI/tests need no display)."""

from __future__ import annotations

import json
import queue
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from .demo import running_sample_simulators
from .errors import ReplayError, SnapshotError, SpecValidationError
from .merge import canonical_json
from .service import DiagnosticService


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return canonical_json(value)
    return str(value)


class DiagnosticsApp:
    def __init__(self, root: Any, service: DiagnosticService, *, spec_path: str | Path):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.messagebox = messagebox
        self.ttk = ttk
        self.root = root
        self.service = service
        self.spec_path = Path(spec_path)
        self.latest_view: dict[str, Any] | None = None
        self.busy = False
        self.closed = False
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        root.title("云原生配置生效结果账本与观测合并诊断台")
        root.geometry("1380x840")
        root.minsize(1080, 680)

        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Warn.TLabel", foreground="#9c2f00")
        style.configure("Ok.TLabel", foreground="#1f5f1f")

        header = ttk.Frame(root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, text="配置生效账本 · 多端观测合并", style="Title.TLabel").pack(
            side="left"
        )
        self.header_label = ttk.Label(header, text="尚未载入轮次")
        self.header_label.pack(side="right")

        toolbar = ttk.Frame(root, padding=(10, 0, 10, 8))
        toolbar.pack(fill="x")
        self.run_button = ttk.Button(toolbar, text="运行一轮观测", command=self.run_round)
        self.run_button.pack(side="left")
        self.replay_button = ttk.Button(toolbar, text="重放校验", command=self.replay)
        self.replay_button.pack(side="left", padx=(6, 0))
        self.reload_button = ttk.Button(
            toolbar, text="重新加载声明", command=self.reload_spec
        )
        self.reload_button.pack(side="left", padx=(6, 0))
        self.refresh_button = ttk.Button(toolbar, text="刷新查询", command=self.refresh)
        self.refresh_button.pack(side="left", padx=(6, 0))
        self.status_label = ttk.Label(toolbar, text="就绪")
        self.status_label.pack(side="right")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.conclusion_tree = self._make_tree_tab(
            notebook,
            "逐键结论",
            (
                ("key", "键", 190),
                ("status", "状态", 150),
                ("expected", "期望值", 135),
                ("effective", "生效值", 135),
                ("quorum", "口径/实际", 100),
                ("divergent", "分歧", 60),
                ("basis", "仲裁依据", 120),
                ("attributions", "归因", 230),
            ),
        )
        self.comparison_tree = self._make_tree_tab(
            notebook,
            "节点对照",
            (
                ("key", "键", 160),
                ("node", "节点", 90),
                ("status", "请求状态", 115),
                ("value", "生效值", 120),
                ("seq", "观测序号", 80),
                ("timestamp", "时间戳ms", 105),
                ("adopted", "采纳", 55),
                ("candidate", "最新候选", 70),
                ("reason", "理由", 320),
            ),
        )
        self.arbitration_tree = self._make_tree_tab(
            notebook,
            "分歧仲裁",
            (
                ("key", "键", 160),
                ("candidate", "最新候选", 160),
                ("winner", "仲裁胜出", 150),
                ("basis", "规则", 130),
                ("majority", "多数计数", 80),
                ("tied", "并列值", 180),
                ("expected", "期望匹配", 80),
                ("text", "依据", 300),
            ),
        )
        self.latency_tree = self._make_tree_tab(
            notebook,
            "延迟统计",
            (
                ("kind", "类别", 80),
                ("scope", "范围", 80),
                ("node", "节点", 90),
                ("count", "样本数", 70),
                ("mean", "均值us", 90),
                ("p95", "P95us", 90),
                ("max", "最大us", 90),
                ("bound", "量化误差上界", 150),
            ),
        )
        ledger_tab = ttk.Frame(notebook)
        notebook.add(ledger_tab, text="账本/版本")
        self.ledger_text = tk.Text(ledger_tab, wrap="none", font=("Consolas", 10))
        ledger_scroll_y = ttk.Scrollbar(
            ledger_tab, orient="vertical", command=self.ledger_text.yview
        )
        ledger_scroll_x = ttk.Scrollbar(
            ledger_tab, orient="horizontal", command=self.ledger_text.xview
        )
        self.ledger_text.configure(
            yscrollcommand=ledger_scroll_y.set, xscrollcommand=ledger_scroll_x.set
        )
        self.ledger_text.grid(row=0, column=0, sticky="nsew")
        ledger_scroll_y.grid(row=0, column=1, sticky="ns")
        ledger_scroll_x.grid(row=1, column=0, sticky="ew")
        ledger_tab.rowconfigure(0, weight=1)
        ledger_tab.columnconfigure(0, weight=1)

        self.footer_label = ttk.Label(root, text="", padding=(10, 0, 10, 8))
        self.footer_label.pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        root.after(80, self._poll_worker)

    def _make_tree_tab(
        self, notebook: Any, title: str, columns: tuple[tuple[str, str, int], ...]
    ) -> Any:
        from tkinter import ttk

        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        tree = ttk.Treeview(
            frame, columns=[item[0] for item in columns], show="headings", selectmode="browse"
        )
        for key, heading, width in columns:
            tree.heading(key, text=heading)
            tree.column(key, width=width, minwidth=45, stretch=(key in {"reason", "text"}))
        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for button in (
            self.run_button,
            self.replay_button,
            self.reload_button,
            self.refresh_button,
        ):
            button.configure(state=state)
        self._set_status(text)

    def _clear_tree(self, tree: Any) -> None:
        children = tree.get_children("")
        if children:
            tree.delete(*children)

    def refresh(self) -> None:
        if self.busy:
            return
        try:
            view = self.service.query_dashboard()
        except Exception as exc:
            self.messagebox.showerror("查询失败", f"{type(exc).__name__}: {exc}")
            self._set_status("查询失败")
            return
        self.latest_view = view
        self._render(view)
        query = view["query_latency"]
        self._set_status(
            f"查询 {query['duration_fixed6_ms']} (1e-6 ms 定点) · "
            f"预算 {query['budget_ms']} ms"
        )

    def _render(self, view: dict[str, Any]) -> None:
        meta = view["meta"]
        round_info = view.get("round")
        digest = meta["current_content_digest"]
        self.header_label.configure(
            text=(
                f"格式 v{meta['format_version']} · 参数 v{meta['parameter_version']} · "
                f"轮次 {(round_info or {}).get('round_id', '无')[:12] or '无'} · "
                f"摘要 {digest[:16]}"
            )
        )
        if view.get("unknown_attempts"):
            self.footer_label.configure(
                text="存在 UNKNOWN 未提交轮次：只能恢复已完整提交的轮次，必须重跑。",
                style="Warn.TLabel",
            )
        elif meta["current_content_digest"] == "INVALID":
            self.footer_label.configure(
                text="账本快照摘要无效，当前视图已整体替换为空，必须重跑。",
                style="Warn.TLabel",
            )
        elif view["ui_latency"].get("last_over_budget"):
            self.footer_label.configure(
                text=(
                    "最近查询超出延迟预算：+" + str(view["ui_latency"]["last_overrun_us"]) + " us；"
                    "所有样本仍保留在聚合中。"
                ),
                style="Warn.TLabel",
            )
        else:
            self.footer_label.configure(
                text=(
                    "延迟真值来源：SQLite 逐次 latency_samples / ui_latency_samples；"
                    "均值与 P95 为整数微秒，P95 为最近秩。"
                ),
                style="Ok.TLabel",
            )

        self._clear_tree(self.conclusion_tree)
        for row in view["conclusions"]:
            self.conclusion_tree.insert(
                "",
                "end",
                values=(
                    row["key"],
                    row["status"],
                    _text(row["expected"]),
                    _text(row["effective_value"]),
                    f"{row['quorum_bp']} / {row['valid_node_count']}/{row['total_nodes']}",
                    _text(row["divergent"]),
                    row.get("winner_basis", ""),
                    ",".join(row.get("attributions", [])),
                ),
            )

        self._clear_tree(self.comparison_tree)
        for row in view["comparisons"]:
            self.comparison_tree.insert(
                "",
                "end",
                values=(
                    row["key"],
                    row["node_id"],
                    row["request_status"],
                    _text(row["value"]),
                    _text(row["observation_seq"]),
                    _text(row["timestamp_ms"]),
                    _text(row["adopted"]),
                    _text(row["latest_candidate"]),
                    row["reason"],
                ),
            )

        self._clear_tree(self.arbitration_tree)
        for row in view["conclusions"]:
            if not row.get("divergent"):
                continue
            candidate = row.get("candidate") or {}
            self.arbitration_tree.insert(
                "",
                "end",
                values=(
                    row["key"],
                    _text(candidate.get("value")),
                    _text(row.get("arbitrated_value")),
                    row.get("winner_basis", ""),
                    row.get("majority_count", ""),
                    _text(row.get("tied_values", [])),
                    _text(row.get("expected_matches")),
                    row.get("basis_text", ""),
                ),
            )

        self._clear_tree(self.latency_tree)
        for row in view["fetch_latency"]:
            self.latency_tree.insert(
                "",
                "end",
                values=(
                    "fetch",
                    row.get("scope", ""),
                    row.get("node_id") or "整体",
                    row["sample_count"],
                    row["mean_us"],
                    row["p95_us"],
                    row["max_us"],
                    row["quantization_error_bound_fixed6_ms"],
                ),
            )
        ui = view["ui_latency"]
        self.latency_tree.insert(
            "",
            "end",
            values=(
                "ui_query",
                "overall",
                "整体",
                ui["sample_count"],
                ui["mean_us"],
                ui["p95_us"],
                ui["max_us"],
                ui["quantization_error_bound_fixed6_ms"],
            ),
        )

        ledger_payload = {
            "meta": view["meta"],
            "round": view["round"],
            "cursor": view["cursor"],
            "parameter_versions": view["parameter_versions"],
            "replacement_events": view["replacement_events"],
            "unknown_attempts": view["unknown_attempts"],
            "query_latency": view["query_latency"],
            "ui_latency": view["ui_latency"],
        }
        self.ledger_text.delete("1.0", "end")
        self.ledger_text.insert(
            "1.0", json.dumps(ledger_payload, ensure_ascii=False, indent=2, sort_keys=True)
        )

    def _run_worker(
        self, operation: Callable[[], Any], busy_text: str, done_text: str
    ) -> None:
        if self.busy:
            return
        self._set_busy(True, busy_text)

        def target() -> None:
            try:
                operation()
            except Exception as exc:
                self.worker_queue.put(("error", exc))
            else:
                self.worker_queue.put(("ok", done_text))

        threading.Thread(target=target, name="diagnostics-worker", daemon=True).start()

    def _poll_worker(self) -> None:
        if self.closed:
            return
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "ok":
                    self._worker_done(str(payload))
                else:
                    self._worker_error(payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_worker)

    def _worker_done(self, text: str) -> None:
        self._set_busy(False, text)
        self.refresh()

    def _worker_error(self, exc: Exception) -> None:
        self._set_busy(False, "操作失败")
        if isinstance(exc, ReplayError) and exc.first_diff:
            diff = exc.first_diff
            message = (
                f"E_REPLAY\n首个差异：key={diff['key']} node={diff['node_id']} "
                f"field={diff['field']}\nledger={diff['ledger']!r}\nreplay={diff['replay']!r}"
            )
        else:
            code = getattr(exc, "code", type(exc).__name__)
            message = f"{code}: {exc}"
        self.messagebox.showerror("诊断台错误", message)

    def run_round(self) -> None:
        self._run_worker(self.service.run_round, "正在并行拉取并合并…", "新一轮已提交")

    def replay(self) -> None:
        self._run_worker(self.service.replay, "正在重放并逐项比较…", "重放校验一致")

    def reload_spec(self) -> None:
        def operation() -> None:
            self.service.reload_spec(self.spec_path)

        self._run_worker(operation, "正在校验并整体替换参数…", "参数版本已刷新")

    def close(self) -> None:
        if self.busy:
            if not self.messagebox.askyesno("退出", "后台任务仍在运行，确定退出？"):
                return
        self.closed = True
        try:
            self.service.close()
        finally:
            self.root.destroy()


def run_ui(
    *,
    spec_path: str | Path,
    ledger_path: str | Path = "runtime/ledger.sqlite",
    log_path: str | Path = "runtime/logs/structured.jsonl",
    sample: str | None = None,
    runtime_dir: str | Path = "runtime/gui",
) -> int:
    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit(
            "Tkinter is unavailable in this Python installation. Use the standard "
            "Windows Python 3.11+ build, which includes Tkinter, or run the CLI."
        )

    runtime = Path(runtime_dir)
    if sample:
        context = running_sample_simulators(sample=sample, runtime_dir=runtime / sample)
    else:
        context = nullcontext(None)
    with context as sample_context:
        effective_spec = spec_path
        if sample_context is not None:
            effective_spec, _scenario, _spec_data = sample_context
        service = DiagnosticService(
            effective_spec, ledger_path=ledger_path, log_path=log_path
        )
        root = tk.Tk()
        try:
            DiagnosticsApp(root, service, spec_path=effective_spec)
            root.mainloop()
        finally:
            try:
                service.close()
            except Exception:
                pass
    return 0
