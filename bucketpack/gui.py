"""Optional Tkinter desktop inspector.

The Tk imports live in this module so headless environments can continue to
use the CLI, tests and the Tk-independent view model.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from .errors import E_GUI, fail
from .view import collect_view_model


def _make_tree(parent: ttk.Frame, columns: list[tuple[str, str, int]]) -> ttk.Treeview:
    tree = ttk.Treeview(parent, columns=[item[0] for item in columns], show="headings")
    for key, heading, width in columns:
        tree.heading(key, text=heading)
        tree.column(key, width=width, stretch=True, anchor="w")
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return tree


def _render(model: dict[str, Any], root: tk.Tk) -> None:
    summary = model["summary"]
    header = ttk.Label(
        root,
        text=(
            f"spec={summary['spec_digest'][:12]}  content={summary['content_digest'][:12]}  "
            f"packs={summary['pack_count']}  rows={summary['row_count']}  "
            f"max_width={summary['max_active_width']}/{summary['global_width_budget']}  "
            f"cost={summary['cost_total_micros']} ({'within' if summary['cost_within_budget'] else 'over'} budget)"
        ),
        anchor="w",
    )
    header.pack(fill="x", padx=8, pady=6)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    dep_frame = ttk.Frame(notebook)
    notebook.add(dep_frame, text="桶列依赖边")
    dep_tree = _make_tree(
        dep_frame,
        [("from", "上游桶列", 260), ("to", "下游桶列", 260)],
    )
    for edge in model["dependencies"]:
        dep_tree.insert("", "end", values=(edge["from"], edge["to"]))

    inv_frame = ttk.Frame(notebook)
    notebook.add(inv_frame, text="本轮失效")
    inv_top = ttk.Frame(inv_frame)
    inv_top.pack(fill="x")
    invalidation = model["invalidation"]
    ttk.Label(
        inv_top,
        text=(
            f"mode={invalidation.get('mode')}  buckets={len(invalidation.get('bucket_columns', []))}  "
            f"packs={len(invalidation.get('packs', []))}"
        ),
        anchor="w",
    ).pack(fill="x", padx=4, pady=4)
    inv_tree = _make_tree(
        inv_frame,
        [
            ("bucket", "失效桶列", 260),
            ("deleted", "已删除", 80),
            ("pack", "受影响包", 260),
            ("trace", "传播轨迹", 420),
        ],
    )
    traces_by_bucket: dict[str, list[str]] = {}
    for item in invalidation.get("trace", []):
        traces_by_bucket.setdefault(item["bucket_column"], []).append(
            f"{item['trigger']}@{item['depth']}"
        )
    affected_by_source: dict[str, list[str]] = {}
    for pack_id in invalidation.get("packs", []):
        source = pack_id.split(":", 1)[0]
        affected_by_source.setdefault(source, []).append(pack_id)
    for bucket_id in sorted(invalidation.get("bucket_columns", [])):
        deleted = bucket_id in invalidation.get("deleted_bucket_columns", [])
        source = bucket_id.split(".", 1)[0]
        inv_tree.insert(
            "",
            "end",
            values=(
                bucket_id,
                "yes" if deleted else "no",
                ",".join(sorted(affected_by_source.get(source, []))),
                ";".join(sorted(traces_by_bucket.get(bucket_id, []))),
            ),
        )

    pack_frame = ttk.Frame(notebook)
    notebook.add(pack_frame, text="包分布与列宽")
    pack_tree = _make_tree(
        pack_frame,
        [
            ("pack_id", "包标识", 155),
            ("source", "数据源", 100),
            ("rows", "行区间", 110),
            ("width", "列宽占用", 100),
            ("cost", "成本(µ)", 90),
            ("wait", "排队ms", 90),
            ("start", "开始ms", 110),
            ("end", "结束ms", 110),
        ],
    )
    for pack in model["packs"]:
        pack_tree.insert(
            "",
            "end",
            values=(
                pack["pack_id"],
                pack["source_id"],
                f"{pack['row_start']}..{pack['row_end']}",
                pack["width_used"],
                pack["cost_micros"],
                pack["wait_ms"],
                pack["start_ms"],
                pack["end_ms"],
            ),
        )

    ledger_frame = ttk.Frame(notebook)
    notebook.add(ledger_frame, text="账本批次与业务码")
    ledger_top = ttk.Frame(ledger_frame)
    ledger_top.pack(fill="x")
    ledger = model["ledger"]
    ttk.Label(
        ledger_top,
        text=(
            f"format={ledger.get('format_version')}  digest={str(ledger.get('content_digest'))[:16]}  "
            f"batches={len(ledger.get('batches', []))}"
        ),
        anchor="w",
    ).pack(fill="x", padx=4, pady=4)
    split = ttk.Panedwindow(ledger_frame, orient="horizontal")
    split.pack(fill="both", expand=True)
    left = ttk.Frame(split)
    right = ttk.Frame(split)
    split.add(left, weight=3)
    split.add(right, weight=2)
    left_tree = _make_tree(
        left,
        [
            ("batch", "批次", 180),
            ("status", "状态", 90),
            ("format", "版本", 60),
            ("committed", "已提交", 70),
            ("unknown", "未知", 70),
        ],
    )
    for batch in ledger.get("batches", []):
        left_tree.insert(
            "",
            "end",
            values=(
                batch["batch_id"],
                batch["status"],
                batch["format_version"],
                batch["committed_count"],
                batch["unknown_count"],
            ),
        )
    right_tree = _make_tree(
        right,
        [
            ("seq", "序号", 60),
            ("code", "业务码", 110),
            ("level", "级别", 70),
            ("message", "消息", 320),
        ],
    )
    for record in model["codes"]:
        right_tree.insert(
            "",
            "end",
            values=(record["seq"], record["code"], record["level"], record["message"]),
        )


def launch_gui(
    *,
    spec_path: str,
    changes_path: str | None,
    ledger_path: str | None,
    log_path: str | None,
) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        fail(E_GUI, "gui.display", "no desktop session is available for Tkinter", error=str(exc))
    root.title("BucketPack 样本特征分桶打包检查器")
    root.geometry("1280x760")
    try:
        model = collect_view_model(
            spec_path=spec_path,
            changes_path=changes_path,
            ledger_path=ledger_path,
            log_path=log_path,
        )
        _render(model, root)
    except Exception:
        root.destroy()
        raise
    root.mainloop()
