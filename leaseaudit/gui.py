from __future__ import annotations

from pathlib import Path

from .errors import BusinessError, ErrorCode
from .view import build_audit_view


def run_gui(ledger_path: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - exercised on Windows GUI hosts
        raise BusinessError(
            ErrorCode.E_ENV,
            "tkinter is unavailable in this Python installation",
            field_path="python.tkinter",
            details={"error": str(exc)},
        ) from exc

    root = tk.Tk()
    root.title("Lease Audit - 并发租约与配额回收审计")
    root.geometry("1380x820")
    root.minsize(1100, 640)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Treeview", rowheight=23)
    style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))

    header_frame = ttk.Frame(root, padding=(10, 8))
    header_frame.pack(fill="x")
    title = ttk.Label(header_frame, text="并发租约与配额回收审计", style="Header.TLabel")
    title.pack(side="left")
    meta_label = ttk.Label(header_frame, text="")
    meta_label.pack(side="left", padx=(18, 0))
    unknown_label = ttk.Label(header_frame, text="", foreground="#a1260d")
    unknown_label.pack(side="right")

    toolbar = ttk.Frame(root, padding=(10, 0, 10, 8))
    toolbar.pack(fill="x")
    refresh_button = ttk.Button(toolbar, text="只读刷新")
    refresh_button.pack(side="left")
    path_label = ttk.Label(toolbar, text=str(Path(ledger_path).resolve()))
    path_label.pack(side="left", padx=(12, 0))

    split = ttk.Panedwindow(root, orient="vertical")
    split.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    source_frame = ttk.LabelFrame(split, text="来源状态（按 source_id 稳定排序）", padding=6)
    event_frame = ttk.LabelFrame(split, text="账本事件（按 seq 稳定排序）", padding=6)
    split.add(source_frame, weight=2)
    split.add(event_frame, weight=3)

    source_columns = (
        "source",
        "share",
        "reserve",
        "retained",
        "active",
        "watermark",
        "reclaimed",
        "unknown",
        "expiries",
        "basis",
    )
    source_tree = ttk.Treeview(source_frame, columns=source_columns, show="headings", height=8)
    source_headings = {
        "source": "来源",
        "share": "并发租约份额",
        "reserve": "保留配额",
        "retained": "当前保留额度",
        "active": "有效授予",
        "watermark": "水位比例",
        "reclaimed": "已回收",
        "unknown": "未知租约",
        "expiries": "到期时刻(ms)",
        "basis": "最近回收依据",
    }
    widths = {
        "source": 110,
        "share": 95,
        "reserve": 85,
        "retained": 100,
        "active": 85,
        "watermark": 90,
        "reclaimed": 75,
        "unknown": 80,
        "expiries": 230,
        "basis": 420,
    }
    for column in source_columns:
        source_tree.heading(column, text=source_headings[column])
        source_tree.column(column, width=widths[column], anchor="w", stretch=column in {"basis", "expiries"})
    source_scroll_y = ttk.Scrollbar(source_frame, orient="vertical", command=source_tree.yview)
    source_scroll_x = ttk.Scrollbar(source_frame, orient="horizontal", command=source_tree.xview)
    source_tree.configure(yscrollcommand=source_scroll_y.set, xscrollcommand=source_scroll_x.set)
    source_tree.grid(row=0, column=0, sticky="nsew")
    source_scroll_y.grid(row=0, column=1, sticky="ns")
    source_scroll_x.grid(row=1, column=0, sticky="ew")
    source_frame.rowconfigure(0, weight=1)
    source_frame.columnconfigure(0, weight=1)

    event_columns = ("seq", "mono", "type", "source", "lease", "amount", "decision", "basis", "digest")
    event_tree = ttk.Treeview(event_frame, columns=event_columns, show="headings", height=13)
    event_headings = {
        "seq": "事件序号",
        "mono": "单调毫秒",
        "type": "事件",
        "source": "来源",
        "lease": "租约",
        "amount": "数量",
        "decision": "裁决",
        "basis": "裁决依据",
        "digest": "归属摘要",
    }
    event_widths = {
        "seq": 80,
        "mono": 120,
        "type": 145,
        "source": 100,
        "lease": 180,
        "amount": 65,
        "decision": 120,
        "basis": 420,
        "digest": 120,
    }
    for column in event_columns:
        event_tree.heading(column, text=event_headings[column])
        event_tree.column(column, width=event_widths[column], anchor="w", stretch=column in {"basis"})
    event_scroll_y = ttk.Scrollbar(event_frame, orient="vertical", command=event_tree.yview)
    event_scroll_x = ttk.Scrollbar(event_frame, orient="horizontal", command=event_tree.xview)
    event_tree.configure(yscrollcommand=event_scroll_y.set, xscrollcommand=event_scroll_x.set)
    event_tree.grid(row=0, column=0, sticky="nsew")
    event_scroll_y.grid(row=0, column=1, sticky="ns")
    event_scroll_x.grid(row=1, column=0, sticky="ew")
    event_frame.rowconfigure(0, weight=1)
    event_frame.columnconfigure(0, weight=1)

    def refresh() -> None:
        # This path is strictly read-only; refreshing never appends ledger events.
        view = build_audit_view(ledger_path)
        meta_label.configure(
            text=(
                f"格式版本 {view['format_version']} | 租约参数版本 {view['params_version']} | "
                f"容量 {view['cluster_capacity']} | 已保留 {view['retained_total']} | "
                f"全局可分配 {view['global_allocatable']} | 总授予 {view['total_active']} | "
                f"事件 {view['event_count']} | digest {view['content_digest'][:16]}…"
            )
        )
        unknown_count = len(view["unknown_leases"])
        unknown_label.configure(
            text=f"未知租约 {unknown_count} 个；需重签发" if unknown_count else "未知租约 0 个"
        )
        source_tree.delete(*source_tree.get_children())
        for row in view["sources"]:
            source_tree.insert(
                "",
                "end",
                iid=row["source_id"],
                values=(
                    row["source_id"],
                    row["lease_share"],
                    row["reserve_quota"],
                    row["retained_quota"],
                    row["active_granted"],
                    row["watermark_percent"],
                    row["reclaimed_total"],
                    "是" if row["unknown_marker"] else "否",
                    ",".join(str(value) for value in row["active_expiries_ms"]) or "-",
                    row["last_reclaim_basis"] or "-",
                ),
            )
        event_tree.delete(*event_tree.get_children())
        for row in view["events"]:
            event_tree.insert(
                "",
                "end",
                iid=f"evt-{row['seq']}",
                values=(
                    row["seq"],
                    row["mono_ms"],
                    row["event_type"],
                    row["source_id"],
                    row["lease_id"],
                    row["amount"],
                    row["decision"],
                    row["basis"],
                    row["ownership_digest"][:16],
                ),
            )

    refresh_button.configure(command=refresh)
    refresh()
    root.mainloop()
