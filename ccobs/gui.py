"""Tkinter 诊断界面。

视图模型 build_view_model 为纯函数, 渲染顺序稳定(窗口按序号、
观测项按名称排序), 可在无显示环境下直接测试; Tk 渲染仅在
run_gui 中延迟导入, 目标平台为 Windows 官方 Python 自带的 Tkinter。
"""
from . import fixedpoint as fp

ITEM_COLUMNS = ("name", "value", "error_bound", "batch_lo", "batch_hi",
                "affected", "halted", "version")


def build_view_model(ledger, sealed_stale=None, snapshots=None, replay_status=""):
    """生成稳定顺序的视图模型(全部为可序列化的基础类型)。"""
    sealed_stale = sealed_stale or set()
    windows = []
    for window in ledger.all_windows():
        items = []
        for name in sorted(window.items):
            item = window.items[name]
            items.append({
                "name": item.name,
                "value": item.value,
                "error_bound": item.error_bound,
                "batch_lo": item.batch_lo,
                "batch_hi": item.batch_hi,
                "affected": item.affected,
                "halted": item.halted,
                "version": item.version,
            })
        windows.append({
            "index": window.index,
            "batch_lo": window.batch_lo,
            "batch_hi": window.batch_hi,
            "sealed": window.sealed,
            "stale": window.stale or window.index in sealed_stale,
            "version": window.version,
            "digest": window.digest,
            "items": items,
        })
    return {
        "config_version": ledger.config_version,
        "windows": windows,
        "sealed": [w["index"] for w in windows if w["sealed"] and not w["stale"]],
        "stale_windows": sorted(sealed_stale |
                                {w["index"] for w in windows if w["stale"]}),
        "anomalies": [a.render() for a in ledger.anomalies],
        "snapshots": list(snapshots or []),
        "replay_status": replay_status,
    }


def run_gui(model, on_replay=None):
    """渲染诊断界面; 仅在调用时导入 tkinter。"""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("CCObs 拥塞控制观测诊断")

    top = ttk.Frame(root)
    top.pack(fill="x", padx=6, pady=4)
    ttk.Label(top, text="配置版本: %d" % model["config_version"]).pack(side="left")
    status = tk.StringVar(value=model["replay_status"])
    ttk.Label(top, textvariable=status).pack(side="right")

    tree = ttk.Treeview(root, columns=ITEM_COLUMNS, show="tree headings", height=18)
    tree.heading("#0", text="观测窗口")
    for col in ITEM_COLUMNS:
        tree.heading(col, text=col)
        tree.column(col, width=90, anchor="e")
    tree.pack(fill="both", expand=True, padx=6)

    def fill():
        tree.delete(*tree.get_children())
        for window in model["windows"]:
            if window["stale"]:
                continue  # 陈旧窗口不参与展示
            label = "窗口%d [%d..%d] %s v%d" % (
                window["index"], window["batch_lo"], window["batch_hi"],
                "已封存" if window["sealed"] else "当前", window["version"])
            node = tree.insert("", "end", text=label, values=("",) * len(ITEM_COLUMNS))
            for item in window["items"]:
                tree.insert(node, "end", text="",
                            values=tuple(item[c] for c in ITEM_COLUMNS))

    fill()

    bottom = ttk.Frame(root)
    bottom.pack(fill="x", padx=6, pady=4)
    ttk.Label(bottom, text="陈旧窗口: %s" % (
        ",".join(str(i) for i in model["stale_windows"]) or "无")).pack(side="left")
    ttk.Label(bottom, text="快照: %s" % (
        ",".join(model["snapshots"]) or "无")).pack(side="left", padx=12)

    if model["anomalies"]:
        ttk.Label(root, text="业务码: " + " ; ".join(model["anomalies"])).pack(
            anchor="w", padx=6)

    if on_replay is not None:
        def do_replay():
            status.set(str(on_replay()))
        ttk.Button(bottom, text="重放校验", command=do_replay).pack(side="right")

    root.mainloop()
