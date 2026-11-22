"""Tkinter 诊断工作台。

四个稳定排序的标签页：
1. 区记录（按名称、类型排序）；
2. 缓存条目（来源格式版本、剥离/补齐字段、TTL 剩余、摘要）；
3. 上游应答（u1/u2 格式与一致性结论）；
4. 快照（SQLite 快照列表与版本集合）。
Windows 官方 Python 自带 tkinter；无显示环境时 main 返回 False 并给出提示。
"""
from __future__ import annotations

from pathlib import Path

from .records import export_view


def run_gui(workbench, *, title: str = "DNS 解析与缓存诊断工作台") -> bool:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - 环境相关
        print(f"[gui] tkinter 不可用（Windows 官方 Python 自带）: {exc}")
        return False
    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - 无显示环境
        print(f"[gui] 无法创建窗口（可能无显示环境）: {exc}")
        return False

    root.title(title)
    root.geometry("1080x720")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    zone_tab = ttk.Frame(notebook)
    cache_tab = ttk.Frame(notebook)
    upstream_tab = ttk.Frame(notebook)
    snapshot_tab = ttk.Frame(notebook)
    notebook.add(zone_tab, text="区记录")
    notebook.add(cache_tab, text="缓存条目")
    notebook.add(upstream_tab, text="上游应答")
    notebook.add(snapshot_tab, text="快照")

    def make_tree(parent: ttk.Frame, columns: tuple[str, ...], widths: dict[str, int]):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=widths.get(column, 140), anchor="w")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return tree

    zone_tree = make_tree(
        zone_tab,
        ("name", "qtype", "ttl", "value", "rdata_hex"),
        {"name": 220, "qtype": 70, "ttl": 70, "value": 260, "rdata_hex": 160},
    )
    cache_tree = make_tree(
        cache_tab,
        ("name", "qtype", "source_version", "ttl_remaining", "value", "view_v1_stripped", "view_v3_filled", "digest"),
        {"name": 180, "qtype": 60, "source_version": 90, "ttl_remaining": 90, "value": 180,
         "view_v1_stripped": 200, "view_v3_filled": 200, "digest": 140},
    )
    upstream_tree = make_tree(
        upstream_tab,
        ("txn_id", "qname", "qtype", "format", "consistent", "attempts", "answers"),
        {"txn_id": 80, "qname": 200, "qtype": 60, "format": 70, "consistent": 90, "attempts": 70, "answers": 320},
    )
    snapshot_tree = make_tree(
        snapshot_tab,
        ("snapshot_id", "created_at_ms", "generation", "zone_version", "format_versions", "ledger_digest"),
        {"snapshot_id": 90, "created_at_ms": 130, "generation": 90, "zone_version": 100,
         "format_versions": 160, "ledger_digest": 260},
    )
    status = tk.StringVar(value="就绪")
    ttk.Label(root, textvariable=status, anchor="w").pack(fill="x", padx=8, pady=(0, 6))

    def refresh() -> None:
        for tree in (zone_tree, cache_tree, upstream_tree, snapshot_tree):
            for item_id in tree.get_children():
                tree.delete(item_id)
        for record in sorted(workbench.resolver.zone.records, key=lambda item: (item.name, item.qtype)):
            zone_tree.insert("", "end", values=(record.name, record.qtype, record.ttl, record.value, record.rdata_hex))
        for stored in workbench.ledger.cache().view():
            entry = stored.entry
            now = workbench.clock.now_ms()
            remaining = stored.remaining_ttl_seconds(now)
            v1 = export_view(entry, "v1", remaining_ttl=remaining)
            v3 = export_view(entry, "v3", remaining_ttl=remaining)
            cache_tree.insert(
                "",
                "end",
                values=(
                    entry.name,
                    entry.qtype,
                    entry.source_version,
                    remaining,
                    entry.value,
                    ",".join(v1.stripped_fields),
                    ",".join(v3.filled_fields),
                    entry.content_digest[:16],
                ),
            )
        for report in workbench.engine.reports:
            upstream_tree.insert(
                "",
                "end",
                values=(
                    hex(report.txn_id),
                    report.qname,
                    report.qtype,
                    report.upstream_format or "-",
                    "" if report.consistent is None else report.consistent,
                    report.attempts or "-",
                    report.source,
                ),
            )
        for snapshot in workbench.snapshots():
            snapshot_tree.insert(
                "",
                "end",
                values=(
                    snapshot["snapshot_id"],
                    snapshot["created_at_ms"],
                    snapshot["generation"],
                    snapshot["zone_version"],
                    ",".join(snapshot["format_versions"]),
                    snapshot["ledger_digest"][:24],
                ),
            )
        generation = workbench.ledger.current()
        status.set(
            f"代数={generation.generation} 区版本={generation.zone_version} "
            f"缓存格式版本={','.join(generation.format_versions)} 条目={len(workbench.ledger.cache())}"
        )

    button_bar = ttk.Frame(root)
    button_bar.pack(fill="x", padx=8)
    ttk.Button(button_bar, text="刷新", command=refresh).pack(side="left")
    ttk.Button(button_bar, text="保存快照", command=lambda: (workbench.save_snapshot(), refresh())).pack(side="left", padx=6)
    ttk.Button(button_bar, text="关闭", command=root.destroy).pack(side="right")
    refresh()
    root.mainloop()
    return True


def describe_gui_unavailable(log_path: Path) -> str:
    return (
        "当前环境无法启动 Tkinter 窗口（Linux 缺少 libtk；Windows 官方 Python 自带 tkinter）。\n"
        f"结构化日志与快照仍可通过命令行演示生成：{log_path}"
    )
