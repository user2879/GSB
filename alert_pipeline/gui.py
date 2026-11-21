from __future__ import annotations

from typing import Any

from .config import PipelineConfig
from .errors import PipelineError
from .logging_utils import StructuredLogger
from .storage import LedgerStore
from .util import pretty_json


def _load_view(config: PipelineConfig) -> dict[str, Any]:
    with LedgerStore(config.database_path) as store:
        generation = store.current_generation()
        if generation is None:
            latest = store.latest_generation()
            if latest is None:
                raise PipelineError(
                    "E_STATE",
                    "no generation exists in the SQLite ledger",
                    field="database",
                )
            generation = latest
        aggregates = store.load_current_aggregates(generation.generation_id)
        ledger = store.load_current_ledger(generation.generation_id)
        ledger_by_id = {entry["alert_id"]: entry for entry in ledger}
        alerts = []
        for aggregate in sorted(
            aggregates,
            key=lambda item: (
                item["first_timestamp_ms"],
                item["service_id"],
                item["alert_id"],
            ),
        ):
            entry = ledger_by_id.get(aggregate["alert_id"], {})
            candidates = entry.get("candidate_services", [])
            paths = entry.get("paths", [])
            paths_text = " | ".join(
                " -> ".join(path.get("path", [])) for path in paths
            )
            alerts.append(
                (
                    aggregate["alert_id"],
                    aggregate["service_id"],
                    aggregate["level"],
                    ",".join(f"v{item}" for item in aggregate["source_format_versions"]),
                    str(aggregate["suppressed_count"]),
                    str(aggregate["total_count"]),
                    ",".join(candidates),
                    paths_text,
                )
            )
        batches = []
        timeline = []
        for batch in store.committed_batches(generation.generation_id):
            batches.append(
                (
                    str(batch["batch_index"]),
                    f'{batch["start_cursor"]}-{batch["end_cursor"]}',
                    batch["business_code"],
                    str(batch["normalized_count"]),
                    str(batch["aggregate_count"]),
                    str(batch["peak_global"]),
                    pretty_json(batch["peak_by_service"]).replace("\n", ""),
                    batch["summary_digest"][:16],
                )
            )
            for item in batch["timeline"]:
                timeline.append(
                    (
                        str(batch["batch_index"]),
                        str(item["ordinal"]),
                        item["stage"],
                        item["service_id"],
                        str(item["wait_ms"]),
                        str(item["run_ms"]),
                        item["outcome"],
                    )
                )
        return {
            "generation": generation,
            "alerts": alerts,
            "ledger": ledger,
            "batches": batches,
            "timeline": timeline,
            "unknown_attempts": list(store.unknown_attempts(generation.generation_id)),
        }


def launch_gui(config: PipelineConfig) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise PipelineError(
            "E_STATE",
            "tkinter is not available; install the Windows Python Tcl/Tk component",
            field="tkinter",
            details={"python_executable": __import__("sys").executable},
        ) from exc

    root = tk.Tk()
    root.title("Alert Denoising & Root Cause Ledger")
    root.geometry("1500x880")
    root.minsize(1000, 600)

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    status_var = tk.StringVar(value="loading")
    ttk.Label(
        top,
        text=f"SQLite: {config.database_path}",
        foreground="#334155",
    ).pack(side="left")
    ttk.Label(top, textvariable=status_var, foreground="#0f766e").pack(
        side="right", padx=8
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    alerts_tab = ttk.Frame(notebook)
    ledger_tab = ttk.Frame(notebook)
    timeline_tab = ttk.Frame(notebook)
    batches_tab = ttk.Frame(notebook)
    notebook.add(alerts_tab, text="告警列表")
    notebook.add(ledger_tab, text="关联账本")
    notebook.add(timeline_tab, text="并发时间线")
    notebook.add(batches_tab, text="账本批次")

    alert_columns = (
        "alert_id",
        "service",
        "level",
        "versions",
        "suppressed",
        "total",
        "candidates",
        "paths",
    )
    alert_tree = ttk.Treeview(alerts_tab, columns=alert_columns, show="headings")
    alert_headings = {
        "alert_id": "告警标识",
        "service": "服务",
        "level": "级别",
        "versions": "来源版本",
        "suppressed": "被抑制",
        "total": "总数",
        "candidates": "根因候选",
        "paths": "推导路径",
    }
    alert_widths = {
        "alert_id": 190,
        "service": 100,
        "level": 80,
        "versions": 90,
        "suppressed": 70,
        "total": 60,
        "candidates": 260,
        "paths": 560,
    }
    for column in alert_columns:
        alert_tree.heading(column, text=alert_headings[column])
        alert_tree.column(column, width=alert_widths[column], anchor="w")
    alert_scroll = ttk.Scrollbar(alerts_tab, orient="vertical", command=alert_tree.yview)
    alert_tree.configure(yscrollcommand=alert_scroll.set)
    alert_tree.pack(side="left", fill="both", expand=True)
    alert_scroll.pack(side="right", fill="y")

    ledger_columns = ("alert_id", "rule", "format", "candidates", "digest")
    ledger_tree = ttk.Treeview(ledger_tab, columns=ledger_columns, show="headings")
    for column, heading, width in (
        ("alert_id", "告警标识", 190),
        ("rule", "规则/版本", 180),
        ("format", "格式版本", 100),
        ("candidates", "根因候选", 360),
        ("digest", "内容摘要", 500),
    ):
        ledger_tree.heading(column, text=heading)
        ledger_tree.column(column, width=width, anchor="w")
    ledger_tree.pack(fill="both", expand=True)

    timeline_columns = ("batch", "ordinal", "stage", "service", "wait", "run", "outcome")
    timeline_tree = ttk.Treeview(
        timeline_tab, columns=timeline_columns, show="headings"
    )
    for column, heading, width in (
        ("batch", "批次", 70),
        ("ordinal", "序", 60),
        ("stage", "阶段", 120),
        ("service", "服务", 120),
        ("wait", "等待 ms", 100),
        ("run", "运行 ms", 100),
        ("outcome", "业务码", 100),
    ):
        timeline_tree.heading(column, text=heading)
        timeline_tree.column(column, width=width, anchor="w")
    timeline_tree.pack(fill="both", expand=True)

    batch_columns = (
        "batch",
        "cursor",
        "business",
        "normalized",
        "aggregates",
        "peak_global",
        "peak_services",
        "digest",
    )
    batch_tree = ttk.Treeview(batches_tab, columns=batch_columns, show="headings")
    for column, heading, width in (
        ("batch", "批次", 70),
        ("cursor", "游标", 100),
        ("business", "业务码", 100),
        ("normalized", "归一数", 90),
        ("aggregates", "聚合数", 90),
        ("peak_global", "全局峰值", 100),
        ("peak_services", "每服务峰值", 260),
        ("digest", "摘要", 360),
    ):
        batch_tree.heading(column, text=heading)
        batch_tree.column(column, width=width, anchor="w")
    batch_tree.pack(fill="both", expand=True)

    def refresh() -> None:
        try:
            view = _load_view(config)
        except Exception as exc:
            status_var.set(f"ERROR: {exc}")
            messagebox.showerror("Ledger error", str(exc))
            return
        for tree in (alert_tree, ledger_tree, timeline_tree, batch_tree):
            for item in tree.get_children():
                tree.delete(item)
        for row in view["alerts"]:
            alert_tree.insert("", "end", values=row)
        ledger = sorted(
            view["ledger"],
            key=lambda item: (item["service_id"], item["alert_id"]),
        )
        for entry in ledger:
            ledger_tree.insert(
                "",
                "end",
                values=(
                    entry["alert_id"],
                    f'{entry["rule_id"]}@{entry["rule_version"]}',
                    f'v{entry["format_version"]}',
                    ",".join(entry.get("candidate_services", [])),
                    entry["content_digest"],
                ),
            )
        for row in view["timeline"]:
            timeline_tree.insert("", "end", values=row)
        for row in view["batches"]:
            batch_tree.insert("", "end", values=row)
        generation = view["generation"]
        status_var.set(
            f'generation={generation.generation_id} version={generation.declaration_version} '
            f'status={generation.status} alerts={len(view["alerts"])} '
            f'batches={len(view["batches"])} unknown={len(view["unknown_attempts"])}'
        )

    button_bar = ttk.Frame(root, padding=(8, 0, 8, 8))
    button_bar.pack(fill="x")
    ttk.Button(button_bar, text="刷新", command=refresh).pack(side="left")

    def run_verify() -> None:
        try:
            from .spec import load_dataset
            from .verifier import verify_pipeline

            dataset = load_dataset(config)
            report = verify_pipeline(
                config,
                dataset,
                logger=StructuredLogger(config.log_path),
                write_report=False,
            )
            status_var.set(
                f'VERIFY OK checks={report["alerts_checked"]} '
                f'digest={report["verification_digest"][:16]}'
            )
            messagebox.showinfo(
                "Verification",
                f'OK\naggregates={report["aggregates_checked"]}\n'
                f'ledger={report["ledger_entries_checked"]}',
            )
        except Exception as exc:
            status_var.set(f"VERIFY ERROR: {exc}")
            messagebox.showerror("Verification error", str(exc))

    ttk.Button(button_bar, text="运行独立验证", command=run_verify).pack(
        side="left", padx=8
    )
    refresh()
    root.mainloop()
