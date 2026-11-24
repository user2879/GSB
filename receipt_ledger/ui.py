"""Tkinter 快速查询界面。

界面只读取账本缓存（一次 SELECT），不触发全量重算；无头环境下
``launch_ui`` 抛出带提示的 RuntimeError，CLI 可降级为文本预览。
"""


def launch_ui(ledger, spec):
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - 取决于系统 Tk 库
        raise RuntimeError(f"Tkinter 不可用，请使用 preview 文本命令: {exc}")

    from .preview import preview

    try:
        data = preview(ledger, spec)
    except Exception as exc:
        raise RuntimeError(f"快速视图读取失败: {exc}")

    root = tk.Tk()
    root.title(f"回执账本快速视图 spec_v{spec.spec_version}")
    columns = ("batch_id", "worker_id", "request_seqs", "output_digest")
    tree = ttk.Treeview(root, columns=columns, show="headings", height=16)
    headings = {
        "batch_id": "批次标识",
        "worker_id": "归属工作器",
        "request_seqs": "参与请求序号",
        "output_digest": "输出摘要",
    }
    widths = {"batch_id": 160, "worker_id": 120,
              "request_seqs": 180, "output_digest": 360}
    for col in columns:
        tree.heading(col, text=headings[col])
        tree.column(col, width=widths[col], anchor="w")
    for item in data["batches"]:
        tree.insert("", "end", values=(
            item["batch_id"], item["worker_id"],
            ",".join(map(str, item["request_seqs"])),
            item["output_digest"][:24] + "…",
        ))
    tree.pack(fill="both", expand=True, padx=8, pady=8)
    status = tk.Label(
        root,
        text=f"单次账本读取 {len(data['batches'])} 批，"
             f"耗时 {data['elapsed_ms']} ms（未触发重算）",
        anchor="w",
    )
    status.pack(fill="x", padx=8, pady=(0, 8))
    root.mainloop()

