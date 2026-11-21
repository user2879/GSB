"""界面层：Tkinter 增量缓存视图。

在无显示环境或缺少 tkinter 的机器上导入不失败（GUI 不可用仅在真正打开
窗口时报 E_UI_UNAVAILABLE），并提供等价的终端视图 render_terminal_view，
保证实验在无头环境同样可复现。
"""

import json

from .errors import BenchError

try:  # Windows 10/11 + 官方 Python 3.11 自带 tkinter。
    import tkinter as tk
    from tkinter import ttk
    _TK_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    tk = None
    ttk = None
    _TK_AVAILABLE = False


class UiUnavailable(BenchError):
    code = "E_UI_UNAVAILABLE"


def render_terminal_view(platform):
    """快速路径的纯文本视图，数据与 Tk 视图同源于 platform.cache。"""
    preview = platform.query_preview()
    lines = [f"request={preview['request_id']} elapsed_ms={preview['elapsed_ms']}"]
    for node in preview["nodes"]:
        lines.append(
            f"  {node['node_id']:<12} v={node['config_version']} "
            f"progress={node['progress']:<11} "
            f"digest={(node['summary_digest'] or '-')[:12]} "
            f"elapsed_ms={node['last_elapsed_ms']}")
    return "\n".join(lines)


class BenchGUI:
    def __init__(self, platform):
        if not _TK_AVAILABLE:
            raise UiUnavailable(
                "tkinter 不可用，无法打开图形界面；可改用 render_terminal_view",
                field="tkinter", basis="当前 Python 未安装 Tk 绑定")
        self.platform = platform
        self.root = tk.Tk()
        self.root.title("配置分发实验台 - 增量缓存视图")
        cols = ("node", "version", "progress", "digest", "elapsed_ms")
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings")
        for col, text in zip(cols, ("节点", "版本", "进度", "摘要前缀", "耗时ms")):
            self.tree.heading(col, text=text)
        self.tree.pack(fill="both", expand=True)
        ttk.Button(self.root, text="刷新(界面查询)",
                   command=self.refresh).pack(side="left")
        ttk.Button(self.root, text="权威重算+一致性",
                   command=self.recompute).pack(side="left")
        self.status = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status).pack(side="bottom")
        self.refresh()

    def refresh(self):
        try:
            preview = self.platform.query_preview()
        except Exception as exc:  # 含 E_LATENCY
            self.status.set(str(exc))
            return
        for row in self.tree.get_children():
            self.tree.delete(row)
        for node in preview["nodes"]:
            self.tree.insert("", "end", values=(
                node["node_id"], node["config_version"], node["progress"],
                (node["summary_digest"] or "-")[:12], node["last_elapsed_ms"]))
        self.status.set(f"查询 {preview['request_id']} 耗时 {preview['elapsed_ms']}ms")

    def recompute(self):
        try:
            latest = self.platform.manifest.latest.version
            result = self.platform.compare_paths(latest)
            self.status.set("一致性: " + json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            self.status.set(str(exc))

    def run(self):
        self.root.mainloop()


def open_gui(platform):
    return BenchGUI(platform)
