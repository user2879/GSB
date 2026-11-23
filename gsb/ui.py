"""Tkinter 诊断界面：灰度去向分布、参数版本隔离、归并采纳/否决、
等待时长与账本版本。tkinter 惰性导入，便于无显示环境下导入本模块。

用法: python -m gsb.ui <ledger.db>
"""

from __future__ import annotations

import json
import sys

from .ledger import Ledger


def build_sections(db_path) -> dict:
    """汇总界面所需数据（与 tkinter 解耦，便于测试）。"""
    ledger = Ledger.open(db_path, content_digest=_read_stored_digest(db_path))
    try:
        meta = ledger.meta()
        sections = {
            "ledger_version": {
                "snapshot_version": meta.get("snapshot_version"),
                "ledger_format_version": meta.get("ledger_format_version"),
                "content_digest": meta.get("content_digest", "")[:16] + "...",
            },
            "route_distribution": ledger.route_distribution(),
            "param_version_isolation": [
                (row["batch_id"], row["param_version"], row["n"])
                for row in ledger.param_version_isolation()
            ],
            "merge_decisions": _merge_decisions(ledger),
            "waits": [
                (row["batch_id"], row["backend_id"], row["attempt"],
                 row["wait_end_ms"] - row["wait_start_ms"])
                for row in ledger.all_waits()
            ],
        }
    finally:
        ledger.close()
    return sections


def _read_stored_digest(db_path) -> str:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='content_digest'"
        ).fetchone()
        return row[0] if row else ""
    finally:
        conn.close()


def _merge_decisions(ledger: Ledger) -> list[dict]:
    decisions = []
    for batch_id in ledger.committed_batches():
        conclusion = ledger.conclusion_for(batch_id)
        decisions.append(
            {
                "batch_id": batch_id,
                "adopted_format": conclusion["adopted_format"],
                "param_version": conclusion["param_version"],
                "source_formats": conclusion["source_formats"],
                "rejected": conclusion["rejected"],
            }
        )
    return decisions


def main(db_path) -> None:
    import tkinter as tk
    from tkinter import ttk

    sections = build_sections(db_path)
    root = tk.Tk()
    root.title("GSB 灰度诊断")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    def add_table(title, headers, rows):
        frame = ttk.Frame(notebook)
        tree = ttk.Treeview(frame, columns=headers, show="headings")
        for header in headers:
            tree.heading(header, text=header)
        for row in rows:
            tree.insert("", "end", values=[str(cell) for cell in row])
        tree.pack(fill="both", expand=True)
        notebook.add(frame, text=title)

    version = sections["ledger_version"]
    add_table(
        "账本版本",
        ("项", "值"),
        [(key, value) for key, value in sorted(version.items())],
    )
    add_table(
        "灰度去向分布",
        ("去向", "请求数"),
        sorted(sections["route_distribution"].items()),
    )
    add_table(
        "参数版本隔离",
        ("批次", "参数版本", "请求数"),
        sections["param_version_isolation"],
    )
    merge_rows = []
    for decision in sections["merge_decisions"]:
        merge_rows.append(
            (
                decision["batch_id"],
                f"采纳 {decision['adopted_format']} (v{decision['param_version']})",
                ",".join(decision["source_formats"]),
                json.dumps(decision["rejected"], ensure_ascii=False),
            )
        )
    add_table("归并采纳/否决", ("批次", "采纳", "来源格式集", "否决对照"), merge_rows)
    add_table(
        "等待时长",
        ("批次", "后端", "尝试", "等待毫秒"),
        sections["waits"],
    )
    root.mainloop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gsb_ledger.db")
