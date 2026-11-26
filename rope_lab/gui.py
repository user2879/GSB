"""Tkinter diagnostic workbench for schemes, errors, budgets and injections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import LabEngine
from .generator import initialize_workspace
from .injections import run_injections
from .tk_bootstrap import import_tkinter
from .validation import load_json, load_jsonl


class DiagnosticsApp:
    def __init__(self, tkinter, ttk, root, workspace_root: str | Path = "."):
        self.tk = tkinter
        self.ttk = ttk
        self.root = root
        self.workspace_root = Path(workspace_root)
        self.config_path = self.workspace_root / "configs/lab_config.json"
        self.samples_path = self.workspace_root / "data/samples.jsonl"
        self.injections_path = self.workspace_root / "injections/injections.json"
        self.engine: LabEngine | None = None
        self.injection_report: dict[str, Any] | None = None

        root.title("RoPE 定点推理诊断工作台")
        root.geometry("1280x820")
        self._build_controls()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.trees: dict[str, Any] = {}
        self._build_tabs()
        self.reload_workspace()

    def _build_controls(self) -> None:
        frame = self.ttk.LabelFrame(self.root, text="编辑与增量重算")
        frame.pack(fill="x", padx=8, pady=(8, 0))

        self.status = self.tk.StringVar(value="正在加载...")
        self.ttk.Label(frame, textvariable=self.status, width=42).grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.ttk.Button(frame, text="重新加载", command=self.reload_workspace).grid(row=0, column=1, padx=4)
        self.ttk.Button(frame, text="增量重算", command=self.rerun_current).grid(row=0, column=2, padx=4)

        self.scheme_name = self.tk.StringVar()
        self.scheme_box = self.ttk.Combobox(frame, textvariable=self.scheme_name, width=18, state="readonly")
        self.scheme_box.grid(row=0, column=3, padx=4)
        self.scheme_box.bind("<<ComboboxSelected>>", lambda _event: self._load_scheme_fields())

        self.base_value = self.tk.StringVar()
        self.scale_value = self.tk.StringVar()
        self.train_value = self.tk.StringVar()
        self.fraction_bits = self.tk.StringVar()
        self.ttk.Label(frame, text="基频").grid(row=0, column=4)
        self.ttk.Entry(frame, textvariable=self.base_value, width=10).grid(row=0, column=5)
        self.ttk.Label(frame, text="缩放").grid(row=0, column=6)
        self.ttk.Entry(frame, textvariable=self.scale_value, width=8).grid(row=0, column=7)
        self.ttk.Label(frame, text="训练长度").grid(row=0, column=8)
        self.ttk.Entry(frame, textvariable=self.train_value, width=6).grid(row=0, column=9)
        self.ttk.Label(frame, text="小数位宽").grid(row=0, column=10)
        self.bit_box = self.ttk.Combobox(frame, textvariable=self.fraction_bits, width=5, state="readonly", values=("4", "8", "12", "16"))
        self.bit_box.grid(row=0, column=11)
        self.ttk.Button(frame, text="应用编辑", command=self.apply_edits).grid(row=0, column=12, padx=8)

    def _make_tree(self, title: str, columns: list[tuple[str, int]]) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        tree = self.ttk.Treeview(frame, columns=[name for name, _width in columns], show="headings")
        for name, width in columns:
            tree.heading(name, text=name)
            tree.column(name, width=width, anchor="w")
        yscroll = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.trees[title] = tree

    def _build_tabs(self) -> None:
        self._make_tree("误差矩阵", [("scheme", 150), ("stage", 120), ("sample_count", 90), ("exceeded", 90), ("first", 70), ("max_abs", 140), ("max_rel", 140), ("samples", 260)])
        self._make_tree("外推统计", [("scheme", 150), ("scope", 180), ("sample_count", 90), ("observations", 100), ("score_mean", 160), ("max_deviation", 150), ("max_abs_error", 150)])
        self._make_tree("边界样本", [("scheme", 150), ("sample_id", 100), ("key_index", 80), ("position", 80), ("distance", 90), ("overrun", 80), ("bin", 160), ("score", 140)])
        self._make_tree("待办队列", [("kind", 180), ("sample_id", 120), ("cost_or_code", 160), ("detail", 500)])
        self._make_tree("业务码", [("code", 260), ("count", 100)])
        self._make_tree("注入清单", [("injection_id", 160), ("kind", 220), ("stage", 120), ("hit", 70), ("expected_code", 220), ("isolation", 420)])
        self._make_tree("复用账本", [("stable_key", 420), ("status", 110), ("changed", 80), ("cache_key", 260), ("basis", 260)])

    def reload_workspace(self) -> None:
        if not self.config_path.exists() or not self.samples_path.exists():
            initialize_workspace(self.workspace_root)
        config, errors = load_json(self.config_path)
        samples, sample_errors = load_jsonl(self.samples_path)
        if errors or sample_errors:
            self.status.set(f"加载失败: {errors + sample_errors}")
            return
        self.engine = LabEngine(config, samples)
        self.engine.run(parallel=True)
        if self.injections_path.exists():
            injection_doc, _ = load_json(self.injections_path)
            injections = injection_doc.get("injections", []) if injection_doc else []
            self.injection_report = run_injections(config, samples, injections)
        names = [scheme["name"] for scheme in config["schemes"]]
        self.scheme_box["values"] = names
        if names and not self.scheme_name.get():
            self.scheme_name.set(names[0])
        self._load_scheme_fields()
        self.refresh()

    def _load_scheme_fields(self) -> None:
        if not self.engine:
            return
        name = self.scheme_name.get()
        scheme = next((item for item in self.engine.config["schemes"] if item["name"] == name), None)
        if scheme:
            self.base_value.set(str(scheme["base_frequency"]))
            self.scale_value.set(str(scheme["angle_scale"]))
        self.train_value.set(str(self.engine.config["train_length"]))
        self.fraction_bits.set(str(self.engine.fixed["fraction_bits"]))

    def apply_edits(self) -> None:
        if not self.engine:
            return
        name = self.scheme_name.get()
        try:
            base = float(self.base_value.get())
            scale = float(self.scale_value.get())
            train_length = int(self.train_value.get())
            fraction_bits = int(self.fraction_bits.get())
        except ValueError as exc:
            self.status.set(f"编辑值无效: {exc}")
            return

        params: dict[str, Any] = {}
        scheme = next(item for item in self.engine.config["schemes"] if item["name"] == name)
        if float(scheme["base_frequency"]) != base:
            params["base_frequency"] = base
        if float(scheme["angle_scale"]) != scale:
            params["angle_scale"] = scale
        if params:
            manifest = self.engine.apply_command({"command_id": "GUI-SCHEME", "op": "update_scheme", "name": name, "params": params})
            self.engine.run(manifest, parallel=True)
        if int(self.engine.config["train_length"]) != train_length:
            manifest = self.engine.apply_command({"command_id": "GUI-TRAIN", "op": "set_train_length", "train_length": train_length})
            self.engine.run(manifest, parallel=True)
        if int(self.engine.fixed["fraction_bits"]) != fraction_bits:
            fixed = dict(self.engine.config["fixed_point"])
            fixed["fraction_bits"] = fraction_bits
            manifest = self.engine.apply_command({"command_id": "GUI-BITS", "op": "set_fixed_point", "fixed_point": fixed})
            self.engine.run(manifest, parallel=True)
        self._load_scheme_fields()
        self.refresh()

    def rerun_current(self) -> None:
        if self.engine:
            self.engine.run({}, parallel=True)
            self.refresh()

    def refresh(self) -> None:
        if not self.engine or not self.engine.last_report:
            return
        report = self.engine.last_report
        self.status.set(
            f"v{report['global_version']} | 样本 {len(report['sample_ids'])} | 账本 {report['cache_stats']['ledger_entries']} | "
            f"复用 {report['cache_stats']['reused']} | 重算 {report['cache_stats']['recomputed']}"
        )
        self._fill_matrix(report["scheme_stage_matrix"])
        self._fill_stats(report["stats_records"])
        self._fill_boundary(report["stats_records"])
        self._fill_pending(report)
        self._fill_codes(report["code_counts"])
        self._fill_injections()
        self._fill_ledger(report["ledger"])

    @staticmethod
    def _clear(tree) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _fill_matrix(self, rows: list[dict[str, Any]]) -> None:
        tree = self.trees["误差矩阵"]
        self._clear(tree)
        for row in rows:
            tree.insert("", "end", values=(row["scheme"], row["stage"], row["sample_count"], row["exceeded_count"], row["first_exceeded_count"], f"{row['max_abs_error']:.3g}", f"{row['max_rel_error']:.3g}", ",".join(row["exceeded_samples"])))

    def _fill_stats(self, records: list[dict[str, Any]]) -> None:
        tree = self.trees["外推统计"]
        self._clear(tree)
        for record in records:
            for scope_name, scope in [*record["categories"].items(), *[(f"bin:{name}", value) for name, value in record["bins"].items()]]:
                tree.insert(
                    "",
                    "end",
                    values=(
                        record["scheme"],
                        scope_name,
                        scope["sample_count"],
                        scope["observation_count"],
                        f"{scope['score_mean']:.6g}",
                        f"{scope['score_max_deviation']:.6g}",
                        f"{scope['score_max_abs_error']:.6g}",
                    ),
                )

    def _fill_boundary(self, records: list[dict[str, Any]]) -> None:
        tree = self.trees["边界样本"]
        self._clear(tree)
        for record in records:
            for item in record["boundary_samples"]:
                tree.insert("", "end", values=(record["scheme"], item["sample_id"], item["key_index"], item["position"], item["relative_distance"], item["overrun"], item["bin"], f"{item['score']:.6g}"))

    def _fill_pending(self, report: dict[str, Any]) -> None:
        tree = self.trees["待办队列"]
        self._clear(tree)
        plan = report["scheduler"]
        for sample_id in report["pending_sample_ids"]:
            tree.insert("", "end", values=("pending", sample_id, plan["point_budget"], "等待下一批优先处理"))
        for sample_id in report["quota_exceeded_sample_ids"]:
            tree.insert("", "end", values=("quota_exceeded", sample_id, plan["sample_quota"], "E_POINT_QUOTA_EXCEEDED"))
        for event in plan["events"]:
            tree.insert("", "end", values=(event["code"], event.get("sample_id", ""), event.get("cost", ""), json.dumps(event, ensure_ascii=False)))
        if not plan["events"] and not report["pending_sample_ids"]:
            tree.insert("", "end", values=("complete", "", "", "当前没有待办或配额异常"))

    def _fill_codes(self, counts: dict[str, int]) -> None:
        tree = self.trees["业务码"]
        self._clear(tree)
        for code, count in sorted(counts.items()):
            tree.insert("", "end", values=(code, count))

    def _fill_injections(self) -> None:
        tree = self.trees["注入清单"]
        self._clear(tree)
        if not self.injection_report:
            tree.insert("", "end", values=("未运行", "", "", "", "", "点击重新加载后执行本地注入清单"))
            return
        for result in self.injection_report["results"]:
            isolation = result.get("isolation", {})
            detail = f"failed_tasks={isolation.get('ledger_failed', '-')}, schemes={isolation.get('stats_schemes', [])}"
            tree.insert("", "end", values=(result["injection_id"], result["kind"], result.get("stage"), "YES" if result["hit"] else "NO", result["expected_code"], detail))

    def _fill_ledger(self, ledger: list[dict[str, Any]]) -> None:
        tree = self.trees["复用账本"]
        self._clear(tree)
        for entry in ledger:
            tree.insert(
                "",
                "end",
                values=(
                    entry["stable_key"],
                    entry["cache_status"],
                    "yes" if entry.get("key_changed") else "no",
                    entry["cache_key"],
                    ",".join(entry.get("invalidation_basis", [])),
                ),
            )


def launch_gui(workspace_root: str | Path = ".") -> int:
    tkinter, error = import_tkinter()
    if tkinter is None:
        print(f"Tkinter 不可用: {error}")
        print("在 Ubuntu 上可执行: sudo apt install python3-tk xvfb")
        return 2
    from tkinter import ttk

    root = tkinter.Tk()
    DiagnosticsApp(tkinter, ttk, root, workspace_root)
    root.mainloop()
    return 0


def smoke_gui(workspace_root: str | Path = ".", close_after_ms: int = 500) -> int:
    tkinter, error = import_tkinter()
    if tkinter is None:
        raise RuntimeError(error)
    from tkinter import ttk

    root = tkinter.Tk()
    app = DiagnosticsApp(tkinter, ttk, root, workspace_root)
    root.update_idletasks()
    root.update()
    rows = {title: len(tree.get_children()) for title, tree in app.trees.items()}
    root.after(close_after_ms, root.destroy)
    root.mainloop()
    print(json.dumps({"gui_smoke": "ok", "rows": rows}, ensure_ascii=False))
    return 0
