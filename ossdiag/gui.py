from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .engine import RunOptions, Workbench
from .errors import DiagnosticError


def dashboard_tables(dashboard: dict[str, Any]) -> dict[str, list[tuple[Any, ...]]]:
    """Convert a dashboard snapshot into stable, sorted rows for tests/UI."""

    objects = sorted(dashboard.get("objects", []), key=lambda item: item["object_key"])
    object_rows: list[tuple[Any, ...]] = []
    shard_rows: list[tuple[Any, ...]] = []
    for item in objects:
        fixed = item["fixed_stats"]
        watermark = item["lease_watermark"]
        object_rows.append(
            (
                item["object_key"],
                item["status"],
                item["selected_path"],
                "YES" if item["boundary_case"] else "NO",
                fixed["completion_display"],
                fixed["average_digest_ms_display"],
                fixed["renewal_rate_display"],
                (
                    f"obj {watermark['object_active']}/{watermark['object_limit']} "
                    f"rest {watermark['object_remaining']}; "
                    f"global {watermark['global_active']}/{watermark['global_limit']} "
                    f"rest {watermark['global_remaining']}"
                ),
                item["digest_status"],
                item["unknown_shards"],
                fixed["error_bounds_display"]["object_aggregate"],
            )
        )
        for shard in sorted(item["shards"], key=lambda row: row["shard_index"]):
            shard_rows.append(
                (
                    item["object_key"],
                    shard["shard_index"],
                    f"[{shard['byte_start']},{shard['byte_end']})",
                    shard["status"],
                    f"{shard['progress_cursor']}/{shard['byte_end']}",
                    (shard["content_digest"] or "-")[:16],
                    shard["lease_version"],
                    shard["renew_count"],
                    shard["completed_mono_ms"] if shard["completed_mono_ms"] is not None else "-",
                    "YES" if shard["unknown"] else "NO",
                )
            )
    snapshots = sorted(dashboard.get("ledger", {}).get("snapshots", []), key=lambda item: item["snapshot_id"])
    snapshot_rows = [
        (
            item["snapshot_id"],
            item["reason"],
            item["backup_path"],
            len(dashboard["ledger"].get("replacement_shards", {}).get(str(item["snapshot_id"]), [])),
            item["created_mono_ms"],
        )
        for item in snapshots
    ]
    return {"objects": object_rows, "shards": shard_rows, "snapshots": snapshot_rows}


def launch_gui(workbench: Workbench, *, auto_run: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - depends on the host Tk runtime
        raise RuntimeError(
            "Tkinter 不可用；Windows 官方 Python 3.11 发行版应自带 Tk。"
        ) from exc

    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - depends on display/Tk installation
        raise RuntimeError("Tkinter 无法初始化图形窗口；请检查 Windows 桌面会话与 Tk 安装。") from exc
    app = DiagnosticWindow(root, tk, ttk, workbench)
    if auto_run:
        root.after(150, app.start_run)
    root.mainloop()


class DiagnosticWindow:
    def __init__(self, root: Any, tk: Any, ttk: Any, workbench: Workbench) -> None:
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.workbench = workbench
        self._run_queue: list[tuple[str, Any]] = []
        self._running = False
        workbench.initialize()
        self.root.title("对象存储分片上传诊断工作台")
        self.root.geometry("1500x900")
        self.root.minsize(1100, 680)

        header = ttk.Frame(root, padding=(10, 8))
        header.pack(fill="x")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(header, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(side="left")
        self.scale_var = tk.BooleanVar(value=False)
        self.digest_fault_var = tk.BooleanVar(value=False)
        self.lease_fault_var = tk.BooleanVar(value=False)
        self.snapshot_fault_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(header, text="E_LEASE 注入", variable=self.lease_fault_var).pack(side="left", padx=(16, 0))
        ttk.Checkbutton(header, text="E_DIGEST 注入", variable=self.digest_fault_var).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(header, text="E_SNAPSHOT 注入", variable=self.snapshot_fault_var).pack(side="left", padx=(8, 0))
        self.run_button = ttk.Button(header, text="运行", command=self.start_run)
        self.run_button.pack(side="right")
        ttk.Button(header, text="刷新（只读）", command=self.refresh).pack(side="right", padx=(0, 8))

        self.path_var = tk.StringVar()
        ttk.Label(root, textvariable=self.path_var, padding=(10, 0)).pack(fill="x")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(6, 4))

        object_frame = ttk.Frame(notebook)
        shard_frame = ttk.Frame(notebook)
        snapshot_frame = ttk.Frame(notebook)
        details_frame = ttk.Frame(notebook)
        notebook.add(object_frame, text="对象与误差界")
        notebook.add(shard_frame, text="分片明细")
        notebook.add(snapshot_frame, text="账本快照")
        notebook.add(details_frame, text="路径与事件")

        object_columns = (
            "object_key", "status", "path", "boundary", "completion", "avg_ms", "renew_rate",
            "lease", "digest", "unknown", "error_bound",
        )
        self.object_tree = ttk.Treeview(object_frame, columns=object_columns, show="headings", height=12)
        object_widths = {
            "object_key": 240, "status": 90, "path": 100, "boundary": 70, "completion": 110,
            "avg_ms": 120, "renew_rate": 110, "lease": 240, "digest": 90, "unknown": 70, "error_bound": 110,
        }
        for column in object_columns:
            self.object_tree.heading(column, text=column)
            self.object_tree.column(column, width=object_widths[column], anchor="center" if column != "object_key" else "w")
        self._with_scroll(object_frame, self.object_tree)

        shard_columns = (
            "object_key", "shard", "range", "status", "cursor", "digest", "lease_ver",
            "renews", "completed", "unknown",
        )
        self.shard_tree = ttk.Treeview(shard_frame, columns=shard_columns, show="headings", height=18)
        shard_widths = {
            "object_key": 240, "shard": 60, "range": 180, "status": 90, "cursor": 150,
            "digest": 170, "lease_ver": 80, "renews": 70, "completed": 130, "unknown": 70,
        }
        for column in shard_columns:
            self.shard_tree.heading(column, text=column)
            self.shard_tree.column(column, width=shard_widths[column], anchor="center" if column != "object_key" else "w")
        self._with_scroll(shard_frame, self.shard_tree)

        snapshot_columns = ("id", "reason", "backup", "replaced", "created")
        self.snapshot_tree = ttk.Treeview(snapshot_frame, columns=snapshot_columns, show="headings", height=12)
        for column in snapshot_columns:
            self.snapshot_tree.heading(column, text=column)
        self.snapshot_tree.column("id", width=60, anchor="center")
        self.snapshot_tree.column("reason", width=240)
        self.snapshot_tree.column("backup", width=720)
        self.snapshot_tree.column("replaced", width=90, anchor="center")
        self.snapshot_tree.column("created", width=140, anchor="center")
        self._with_scroll(snapshot_frame, self.snapshot_tree)

        self.details_text = tk.Text(details_frame, wrap="none", font=("Consolas", 10))
        details_scroll = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scroll.set)
        self.details_text.pack(side="left", fill="both", expand=True)
        details_scroll.pack(side="right", fill="y")

        footer = ttk.Frame(root, padding=(10, 4))
        footer.pack(fill="x")
        self.footer_var = tk.StringVar()
        ttk.Label(footer, textvariable=self.footer_var).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.refresh()

    def _with_scroll(self, parent: Any, tree: Any) -> None:
        vertical = self.ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        horizontal = self.ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)

    def refresh(self) -> None:
        try:
            dashboard = self.workbench.build_dashboard()
        except Exception as exc:
            self.status_var.set(f"刷新失败：{exc}")
            return
        tables = dashboard_tables(dashboard)
        for tree, rows in (
            (self.object_tree, tables["objects"]),
            (self.shard_tree, tables["shards"]),
            (self.snapshot_tree, tables["snapshots"]),
        ):
            children = tree.get_children()
            if children:
                tree.delete(*children)
            for row in rows:
                tree.insert("", "end", values=row)
        self.path_var.set(
            f"账本: {dashboard['ledger']['path']} | 格式版本: {dashboard['ledger']['format_version']} | "
            f"摘要: {str(dashboard['ledger']['ledger_digest'])[:16]}… | 账本摘要自洽: "
            f"{dashboard['ledger']['ledger_digest'] == dashboard['ledger']['recomputed_digest']}"
        )
        self.footer_var.set(
            f"定点缩放 10^{dashboard['scale_policy']['scale_exponent']}；截断=向下取整；饱和上限=2^63-1；"
            f"吞吐已提交 {dashboard['throughput']['committed_bytes']}/{dashboard['throughput']['limit_bytes']} 字节"
        )
        self._render_details(dashboard)

    def _render_details(self, dashboard: dict[str, Any]) -> None:
        lines: list[str] = []
        lines.append("路径选择与决胜依据")
        for item in dashboard["objects"]:
            lines.append(
                f"- {item['object_key']}: path={item['selected_path']} boundary={item['boundary_case']} "
                f"reason={item['path_reason']}"
            )
            for estimate in item.get("path_scores", []):
                lines.append(
                    f"    {estimate['path']}: score={estimate['score']} leaves={estimate['leaf_count']} "
                    f"batches={estimate['batch_count']} shards={estimate['shard_count']} "
                    f"memory={estimate['memory_class']}"
                )
        lines.append("")
        lines.append("误差敏感排序")
        for item in dashboard["ranking"]:
            lines.append(
                f"- #{item['rank']} {item['object_key']} completion={item['completion_display']} "
                f"affected={item['ordering_affected_by_error_bound']}"
            )
        if dashboard["ranking_affected_pairs"]:
            lines.append("  受误差界影响的比较对：")
            for pair in dashboard["ranking_affected_pairs"]:
                lines.append(
                    f"  {pair['left']} <-> {pair['right']} delta={pair['completion_delta_fixed']}/"
                    f"{dashboard['scale_policy']['scale']} bound={pair['bound_fixed']}/"
                    f"{dashboard['scale_policy']['scale']}"
                )
        snapshots = dashboard["ledger"].get("snapshots", [])
        if snapshots:
            lines.append("")
            lines.append("账本替换快照（旧账本只读）")
            for snapshot in snapshots:
                replaced = dashboard["ledger"].get("replacement_shards", {}).get(str(snapshot["snapshot_id"]), [])
                replaced_text = ", ".join(f"{item['object_key']}#{item['shard_index']}" for item in replaced)
                lines.append(
                    f"- snapshot={snapshot['snapshot_id']} reason={snapshot['reason']} "
                    f"backup={snapshot['backup_path']} replaced=[{replaced_text}]"
                )
        lines.append("")
        lines.append("刷新为只读快照；运行按钮才写入账本。强制终止退出报告不可用。")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", "\n".join(lines))

    def start_run(self) -> None:
        if self._running:
            return
        self._running = True
        self.run_button.configure(state="disabled")
        self.status_var.set("运行中…")
        options = RunOptions(
            inject_lease_overflow=self.lease_fault_var.get(),
            inject_digest_mismatch=self.digest_fault_var.get(),
            inject_snapshot_conflict=self.snapshot_fault_var.get(),
        )

        def worker() -> None:
            try:
                summary = self.workbench.run(options)
                self._run_queue.append(("ok", summary))
            except BaseException as exc:
                self._run_queue.append(("error", exc))

        threading.Thread(target=worker, daemon=True, name="ossdiag-gui-run").start()
        self.root.after(100, self._poll_run)

    def _poll_run(self) -> None:
        if self._run_queue:
            kind, value = self._run_queue.pop(0)
            self._running = False
            self.run_button.configure(state="normal")
            if kind == "ok":
                summary = value
                codes = sorted({item["code"] for item in summary.errors})
                self.status_var.set(f"运行结束：ok={summary.ok} errors={codes or 'none'}")
                self.refresh()
            else:
                self.status_var.set(f"运行失败：{value}")
            return
        self.root.after(100, self._poll_run)

    def _close(self) -> None:
        if self._running:
            self.status_var.set("运行仍在进行，请等待完成后关闭窗口。")
            return
        try:
            self.workbench.close()
        finally:
            self.root.destroy()
