from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk

from .diagnostics import build_diagnostic_model
from .engine import ArbiterEngine
from .errors import ArbiterError
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .snapshot import restore_workspace, write_snapshot
from .spec import Specs, load_specs
from .util import atomic_write_json, read_json


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


class ArbiterApp(tk.Tk):
    def __init__(
        self,
        *,
        environment_path: str,
        views_path: str,
        db_path: str,
        snapshot_path: str,
        log_path: str,
    ) -> None:
        super().__init__()
        self.title("跨环境配置仲裁一致性诊断")
        self.geometry("1450x880")
        self.minsize(1100, 680)
        self.environment_path = environment_path
        self.views_path = views_path
        self.db_path = db_path
        self.snapshot_path = snapshot_path
        self.logger = StructuredLogger(log_path)
        self.ledger: Ledger | None = None
        self.specs: Specs | None = None
        self.model: dict[str, Any] | None = None
        self._env_rows: dict[str, dict[str, Any]] = {}
        self._chain_rows: dict[str, dict[str, Any]] = {}
        self._authority_rows: dict[str, dict[str, Any]] = {}
        self._error_rows: dict[str, dict[str, Any]] = {}
        self._version_rows: dict[str, dict[str, Any]] = {}
        self._configure_style()
        self._build_shell()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self.refresh_read_only)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Toolbar.TFrame", padding=8)
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Mono.TText", font=("Consolas", 9))

    def _build_shell(self) -> None:
        toolbar = ttk.Frame(self, style="Toolbar.TFrame")
        toolbar.pack(fill="x", side="top")
        ttk.Label(toolbar, text="配置仲裁诊断工作台", style="Title.TLabel").pack(side="left", padx=(0, 16))
        ttk.Button(toolbar, text="只读刷新", command=self.refresh_read_only).pack(side="left", padx=3)
        ttk.Button(toolbar, text="裁决并提交", command=self.run_arbitration).pack(side="left", padx=3)
        ttk.Button(toolbar, text="写入快照", command=self.create_snapshot).pack(side="left", padx=3)
        ttk.Button(toolbar, text="恢复快照", command=self.restore_snapshot).pack(side="left", padx=3)
        self.status_var = tk.StringVar(value="尚未载入")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._build_environment_tab()
        self._build_chain_tab()
        self._build_authority_tab()
        self._build_error_tab()
        self._build_versions_tab()

    @staticmethod
    def _tree(parent: tk.Misc, columns: tuple[str, ...], *, height: int = 20) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=height)
        for column in columns:
            tree.heading(column, text=column)
            width = 120
            if column in {"key", "chain_signature"}:
                width = 230
            elif column in {"source_id", "environment_id"}:
                width = 150
            elif column in {"value_text", "value_fixed"}:
                width = 105
            tree.column(column, width=width, anchor="w", stretch=True)
        return tree

    @staticmethod
    def _with_vertical_scroll(parent: tk.Misc, tree: ttk.Treeview) -> None:
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_environment_tab(self) -> None:
        tab = ttk.Panedwindow(self.notebook, orient="horizontal")
        self.notebook.add(tab, text="环境与来源")
        left = ttk.Frame(tab)
        right = ttk.LabelFrame(tab, text="编辑选中条目", padding=10)
        tab.add(left, weight=4)
        tab.add(right, weight=2)
        self.environment_tree = self._tree(
            left,
            ("environment_id", "depth", "source_id", "source_type", "priority_bias", "key", "value_text", "observed_seq", "captured_at_ms"),
        )
        self.environment_tree.bind("<<TreeviewSelect>>", self._on_environment_select)
        self._with_vertical_scroll(left, self.environment_tree)

        self.edit_vars = {
            "environment_id": tk.StringVar(),
            "depth": tk.StringVar(),
            "source_id": tk.StringVar(),
            "source_type": tk.StringVar(),
            "priority_bias": tk.StringVar(),
            "key": tk.StringVar(),
            "value_text": tk.StringVar(),
            "observed_seq": tk.StringVar(),
            "captured_at_ms": tk.StringVar(),
        }
        fields = [
            ("环境标识", "environment_id", True),
            ("环境深度", "depth", False),
            ("来源标识", "source_id", True),
            ("来源类型", "source_type", True),
            ("优先级偏置", "priority_bias", False),
            ("配置键", "key", True),
            ("取值", "value_text", False),
            ("观测序号", "observed_seq", False),
            ("采集时刻(ms)", "captured_at_ms", False),
        ]
        for row, (label, name, readonly) in enumerate(fields):
            ttk.Label(right, text=label).grid(row=row, column=0, sticky="w", pady=4)
            state = "readonly" if readonly else "normal"
            ttk.Entry(right, textvariable=self.edit_vars[name], state=state).grid(
                row=row, column=1, sticky="ew", pady=4, padx=(8, 0)
            )
        right.columnconfigure(1, weight=1)
        ttk.Button(right, text="应用并写回环境 JSON", command=self.save_environment_edit).grid(
            row=len(fields), column=0, columnspan=2, sticky="ew", pady=(14, 4)
        )
        ttk.Label(
            right,
            text="保存只修改本地环境声明，不触发裁决。随后点击“裁决并提交”。",
            wraplength=360,
        ).grid(row=len(fields) + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_chain_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab, text="逐环境生效链")
        self.chain_tree = self._tree(
            tab,
            ("view_id", "key", "environment_id", "rank", "source_id", "source_type", "priority_bias", "value_text", "value_fixed", "observed_seq", "captured_at_ms", "binding_eligible", "decision"),
        )
        self._with_vertical_scroll(tab, self.chain_tree)

    def _build_authority_tab(self) -> None:
        tab = ttk.Panedwindow(self.notebook, orient="vertical")
        self.notebook.add(tab, text="观测分布 / 权威与否决")
        top = ttk.Frame(tab)
        bottom = ttk.LabelFrame(tab, text="选中键的权威取值、采纳集与被否决理由", padding=6)
        tab.add(top, weight=3)
        tab.add(bottom, weight=2)
        self.authority_tree = self._tree(
            top,
            ("view_id", "key", "environment_id", "effective_source", "source_type", "value_text", "value_fixed", "observed_seq", "captured_at_ms"),
        )
        self.authority_tree.bind("<<TreeviewSelect>>", self._on_authority_select)
        self._with_vertical_scroll(top, self.authority_tree)
        self.authority_text = tk.Text(bottom, wrap="none", height=10, font=("Consolas", 9))
        self.authority_text.pack(fill="both", expand=True)

    def _build_error_tab(self) -> None:
        tab = ttk.Panedwindow(self.notebook, orient="vertical")
        self.notebook.add(tab, text="误差界与归因")
        top = ttk.Frame(tab)
        bottom = ttk.LabelFrame(tab, text="定点误差、比较误差界和稳定决胜归因", padding=6)
        tab.add(top, weight=2)
        tab.add(bottom, weight=3)
        self.error_tree = self._tree(
            top,
            ("view_id", "key", "strategy", "error_influenced", "comparison_bound", "key_aggregation_bound", "view_aggregation_bound", "authority_fixed"),
        )
        self.error_tree.bind("<<TreeviewSelect>>", self._on_error_select)
        self._with_vertical_scroll(top, self.error_tree)
        self.error_text = tk.Text(bottom, wrap="none", height=14, font=("Consolas", 9))
        self.error_text.pack(fill="both", expand=True)

    def _build_versions_tab(self) -> None:
        tab = ttk.Panedwindow(self.notebook, orient="vertical")
        self.notebook.add(tab, text="版本与陈旧传播")
        top = ttk.Frame(tab)
        bottom = ttk.LabelFrame(tab, text="陈旧传播轨迹", padding=6)
        tab.add(top, weight=2)
        tab.add(bottom, weight=2)
        self.version_tree = self._tree(
            top,
            ("view_id", "current_status", "current_version", "history_version", "history_status", "result_sha256", "stale_reason"),
        )
        self.version_tree.bind("<<TreeviewSelect>>", self._on_version_select)
        self._with_vertical_scroll(top, self.version_tree)
        self.version_text = tk.Text(bottom, wrap="none", height=10, font=("Consolas", 9))
        self.version_text.pack(fill="both", expand=True)

    def _close_ledger(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
            self.ledger = None

    def refresh_read_only(self) -> None:
        try:
            specs = load_specs(self.environment_path, self.views_path)
            self._close_ledger()
            ledger = Ledger(self.db_path)
            ledger.verify_integrity()
            model = build_diagnostic_model(specs, ledger)
            self.specs = specs
            self.ledger = ledger
            self.model = model
            self._render_all()
            pending = len(model["unknown_views"]) + len(model["changed_views"])
            self.status_var.set(
                f"只读刷新完成 · 账本 {model['ledger_content_digest'][:12]} · "
                f"已完成 {len(model['completed_views'])} / {len(specs.views)} · 待重跑 {pending}"
            )
        except Exception as exc:
            self._show_error("只读刷新失败", exc)

    def run_arbitration(self) -> None:
        try:
            specs = load_specs(self.environment_path, self.views_path)
            self._close_ledger()
            ledger = Ledger(self.db_path)
            try:
                ledger.verify_integrity()
                report = ArbiterEngine(specs, ledger).run_all()
                snapshot = write_snapshot(self.snapshot_path, specs, ledger)
                self.ledger = ledger
                self.specs = specs
                self.model = build_diagnostic_model(specs, ledger)
            except BaseException:
                ledger.close()
                raise
            self._render_all()
            committed = sum(1 for item in report["views"] if item["action"] == "committed")
            self.status_var.set(
                f"裁决完成 · 提交 {committed} 个视图 · 快照 {snapshot['content_digest'][:12]}"
            )
        except Exception as exc:
            self._show_error("裁决失败", exc)

    def create_snapshot(self) -> None:
        try:
            if self.ledger is None or self.specs is None:
                self.refresh_read_only()
            if self.ledger is None or self.specs is None:
                return
            payload = write_snapshot(self.snapshot_path, self.specs, self.ledger)
            self.status_var.set(f"快照已写入 · {payload['content_digest'][:16]}")
        except Exception as exc:
            self._show_error("写入快照失败", exc)

    def restore_snapshot(self) -> None:
        try:
            self._close_ledger()
            report = restore_workspace(
                snapshot_path=self.snapshot_path,
                environment_path=self.environment_path,
                views_path=self.views_path,
                db_path=self.db_path,
            )
            self.refresh_read_only()
            replay_count = 0 if report.get("replay") is None else report["replay"]["compared_count"]
            self.status_var.set(
                f"快照恢复 {report['mode']} · 重放比较 {replay_count} 个视图 · {report['reason']}"
            )
        except Exception as exc:
            self._close_ledger()
            self._show_error("恢复快照失败", exc)

    def save_environment_edit(self) -> None:
        try:
            document = read_json(self.environment_path)
            environment_id = self.edit_vars["environment_id"].get()
            source_id = self.edit_vars["source_id"].get()
            key = self.edit_vars["key"].get()
            depth = int(self.edit_vars["depth"].get())
            bias = int(self.edit_vars["priority_bias"].get())
            observed_seq = int(self.edit_vars["observed_seq"].get())
            captured_at_ms = int(self.edit_vars["captured_at_ms"].get())
            value = json.loads(self.edit_vars["value_text"].get())
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("取值必须是 JSON 数字")
            changed = False
            for environment in document.get("environments", []):
                if environment.get("environment_id") != environment_id:
                    continue
                environment["depth"] = depth
                for source in environment.get("sources", []):
                    if source.get("source_id") != source_id:
                        continue
                    source["priority_bias"] = bias
                    for observation in source.get("observations", []):
                        if observation.get("key") == key:
                            observation["value"] = value
                            observation["observed_seq"] = observed_seq
                            observation["captured_at_ms"] = captured_at_ms
                            changed = True
                            break
            if not changed:
                raise ValueError("未找到选中的环境/来源/键条目")
            with tempfile.TemporaryDirectory(prefix="arbiter-ui-validate-") as temporary:
                temporary_environment = Path(temporary) / "environments.json"
                atomic_write_json(temporary_environment, document)
                load_specs(temporary_environment, self.views_path)
            atomic_write_json(self.environment_path, document)
            self.refresh_read_only()
            self.status_var.set("环境声明已保存；尚未提交裁决")
        except Exception as exc:
            self._show_error("保存环境编辑失败", exc)

    def _render_all(self) -> None:
        assert self.model is not None
        self._render_environments()
        self._render_chains()
        self._render_authority()
        self._render_errors()
        self._render_versions()

    @staticmethod
    def _clear(tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _render_environments(self) -> None:
        assert self.model is not None and self.specs is not None
        self._clear(self.environment_tree)
        self._env_rows.clear()
        for environment in self.model["environments"]:
            for source_id in sorted(environment["sources"]):
                source = environment["sources"][source_id]
                for key in sorted(source["observations"]):
                    observation = source["observations"][key]
                    row_id = f"{environment['environment_id']}|{source_id}|{key}"
                    self._env_rows[row_id] = {
                        "environment_id": environment["environment_id"],
                        "depth": environment["depth"],
                        "source_id": source_id,
                        "source_type": source["source_type"],
                        "priority_bias": source["priority_bias"],
                        "key": key,
                        "value_text": observation["value_text"],
                        "observed_seq": observation["observed_seq"],
                        "captured_at_ms": observation["captured_at_ms"],
                    }
                    self.environment_tree.insert(
                        "",
                        "end",
                        iid=row_id,
                        values=(
                            environment["environment_id"],
                            environment["depth"],
                            source_id,
                            source["source_type"],
                            source["priority_bias"],
                            key,
                            observation["value_text"],
                            observation["observed_seq"],
                            observation["captured_at_ms"],
                        ),
                    )

    def _render_chains(self) -> None:
        assert self.model is not None
        self._clear(self.chain_tree)
        self._chain_rows.clear()
        for view in self.model["views"]:
            for key_result in view["keys"]:
                for chain in key_result["chains"]:
                    for step in chain["steps"]:
                        row_id = f"{view['view_id']}|{key_result['key']}|{chain['environment_id']}|{step['rank']}"
                        self._chain_rows[row_id] = {"key_result": key_result, "chain": chain, "step": step}
                        self.chain_tree.insert(
                            "",
                            "end",
                            iid=row_id,
                            values=(
                                view["view_id"],
                                key_result["key"],
                                chain["environment_id"],
                                step["rank"],
                                step["source_id"],
                                step["source_type"],
                                step["priority_bias"],
                                step["value_text"],
                                step["value_fixed"],
                                step["observed_seq"],
                                step["captured_at_ms"],
                                "yes" if step["binding_eligible"] else "no",
                                step["decision"],
                            ),
                        )

    def _render_authority(self) -> None:
        assert self.model is not None
        self._clear(self.authority_tree)
        self._authority_rows.clear()
        for view in self.model["views"]:
            for key_result in view["keys"]:
                for observation in key_result["observations"]:
                    row_id = f"{view['view_id']}|{key_result['key']}|{observation['environment_id']}"
                    self._authority_rows[row_id] = {"view": view, "key_result": key_result}
                    self.authority_tree.insert(
                        "",
                        "end",
                        iid=row_id,
                        values=(
                            view["view_id"],
                            key_result["key"],
                            observation["environment_id"],
                            observation["source_id"] or "<none>",
                            observation["source_type"] or "<none>",
                            observation["value_text"] if observation["value_text"] is not None else "<none>",
                            observation["value_fixed"] if observation["value_fixed"] is not None else "<none>",
                            observation["observed_seq"] if observation["observed_seq"] is not None else "<none>",
                            observation["captured_at_ms"] if observation["captured_at_ms"] is not None else "<none>",
                        ),
                    )

    def _render_errors(self) -> None:
        assert self.model is not None
        self._clear(self.error_tree)
        self._error_rows.clear()
        for view in self.model["views"]:
            for key_result in view["keys"]:
                row_id = f"{view['view_id']}|{key_result['key']}"
                self._error_rows[row_id] = {"view": view, "key_result": key_result}
                self.error_tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    values=(
                        view["view_id"],
                        key_result["key"],
                        key_result["strategy"],
                        "yes" if key_result["error_influenced"] else "no",
                        key_result["error_model"]["comparison_error_bound_fixed_units"],
                        key_result["key_aggregation_error_upper_fixed_units"],
                        key_result["view_aggregation_error_upper_bound_fixed_units"],
                        key_result["authoritative"]["value_fixed"],
                    ),
                )

    def _render_versions(self) -> None:
        assert self.model is not None
        self._clear(self.version_tree)
        self._version_rows.clear()
        for view in self.model["views"]:
            if not view["history"]:
                row_id = f"{view['view_id']}|current"
                self._version_rows[row_id] = {"view": view, "history": None}
                self.version_tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    values=(
                        view["view_id"],
                        view["status"],
                        view["version"],
                        "<none>",
                        "requires_rerun",
                        "",
                        view["reason"] or "",
                    ),
                )
            for history in view["history"]:
                row_id = f"{view['view_id']}|{history['version']}"
                self._version_rows[row_id] = {"view": view, "history": history}
                self.version_tree.insert(
                    "",
                    "end",
                    iid=row_id,
                    values=(
                        view["view_id"],
                        view["status"],
                        view["version"],
                        history["version"],
                        history["status"],
                        history["result_sha256"],
                        history["stale_reason"] or "",
                    ),
                )

    def _on_environment_select(self, _event: tk.Event) -> None:
        selection = self.environment_tree.selection()
        if not selection:
            return
        row = self._env_rows[selection[0]]
        for name, value in row.items():
            self.edit_vars[name].set(str(value))

    def _on_authority_select(self, _event: tk.Event) -> None:
        selection = self.authority_tree.selection()
        if not selection:
            return
        row = self._authority_rows[selection[0]]
        key_result = row["key_result"]
        detail = {
            "view_id": row["view"]["view_id"],
            "key": key_result["key"],
            "strategy": key_result["strategy"],
            "authoritative": key_result["authoritative"],
            "accepted": key_result["accepted"],
            "rejected": key_result["rejected"],
            "error_influenced": key_result["error_influenced"],
        }
        self.authority_text.delete("1.0", "end")
        self.authority_text.insert("1.0", _json_text(detail))

    def _on_error_select(self, _event: tk.Event) -> None:
        selection = self.error_tree.selection()
        if not selection:
            return
        row = self._error_rows[selection[0]]
        key_result = row["key_result"]
        detail = {
            "view_id": row["view"]["view_id"],
            "key": key_result["key"],
            "policy": key_result["policy"],
            "error_model": key_result["error_model"],
            "key_aggregation_error_upper_fixed_units": key_result["key_aggregation_error_upper_fixed_units"],
            "view_aggregation_error_upper_bound_fixed_units": key_result["view_aggregation_error_upper_bound_fixed_units"],
            "error_influenced": key_result["error_influenced"],
            "error_attribution": key_result["error_attribution"],
        }
        self.error_text.delete("1.0", "end")
        self.error_text.insert("1.0", _json_text(detail))

    def _on_version_select(self, _event: tk.Event) -> None:
        selection = self.version_tree.selection()
        if not selection:
            return
        row = self._version_rows[selection[0]]
        detail = {
            "view": {
                "view_id": row["view"]["view_id"],
                "current_status": row["view"]["status"],
                "current_version": row["view"]["version"],
                "fingerprint": row["view"]["fingerprint"],
            },
            "ledger_history": row["view"]["history"],
            "stale_findings": row["view"]["stale_findings"],
            "requires_rerun": row["view"]["requires_rerun"],
            "reason": row["view"]["reason"],
        }
        self.version_text.delete("1.0", "end")
        self.version_text.insert("1.0", _json_text(detail))

    def _show_error(self, title: str, exc: Exception) -> None:
        if isinstance(exc, ArbiterError):
            message = f"{exc.code}: {exc.message}\n\n{_json_text(exc.details)}"
            self.logger.log(
                "ui_error",
                level="ERROR",
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
        else:
            message = f"{type(exc).__name__}: {exc}"
            self.logger.log("ui_error", level="ERROR", code="E_INTERNAL", message=str(exc))
        self.status_var.set(title)
        messagebox.showerror(title, message, parent=self)

    def _on_close(self) -> None:
        self._close_ledger()
        self.destroy()


def launch_gui(
    *,
    environment_path: str,
    views_path: str,
    db_path: str,
    snapshot_path: str,
    log_path: str,
) -> None:
    app = ArbiterApp(
        environment_path=environment_path,
        views_path=views_path,
        db_path=db_path,
        snapshot_path=snapshot_path,
        log_path=log_path,
    )
    app.mainloop()
