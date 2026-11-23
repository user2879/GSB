"""Tkinter 诊断界面（只读）：依赖图、失效集合、逐节点回执来源与账本版本。

渲染顺序稳定（拓扑分层 + 字典序）；刷新仅重新读取文件，不写账本。
"""

import os
import sqlite3
import tkinter as tk
from tkinter import ttk

from .graph import topo_order
from .ledger import FORMAT_VERSION
from .spec import load_declaration

LAYER_X = 170
ROW_Y = 70
MARGIN = 60


class DiagnosticsApp(tk.Tk):
    def __init__(self, workdir):
        super().__init__()
        self.workdir = workdir
        self.title("cfgpipe diagnostics - %s" % workdir)
        self.geometry("980x640")
        self.canvas = tk.Canvas(self, bg="white", width=620)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.info = tk.Text(right, width=48, height=40)
        self.info.pack(fill=tk.BOTH, expand=True)
        ttk.Button(right, text="刷新（只读）", command=self.render).pack()
        self.render()

    def _load(self):
        decl = load_declaration(os.path.join(self.workdir, "declaration.json"))
        ledger_path = os.path.join(self.workdir, "ledger.db")
        meta, batches, receipts, invalidation = {}, [], [], []
        if os.path.exists(ledger_path):
            db = sqlite3.connect("file:%s?mode=ro" % ledger_path, uri=True)
            try:
                meta = dict(db.execute("SELECT key, value FROM meta"))
                batches = list(db.execute(
                    "SELECT batch_id, seq, changeset_version FROM batches"
                    " ORDER BY seq"))
                if batches:
                    last = batches[-1][0]
                    receipts = list(db.execute(
                        "SELECT node_id, source, config_version, basis_version"
                        " FROM receipts WHERE batch_id = ? ORDER BY node_id",
                        (last,)))
                    invalidation = list(db.execute(
                        "SELECT step, kind, name, reason FROM invalidation"
                        " WHERE batch_id = ? ORDER BY step", (last,)))
            finally:
                db.close()
        return decl, meta, batches, receipts, invalidation

    def render(self):
        decl, meta, batches, receipts, invalidation = self._load()
        derived = {it.key: it.derived_from for it in decl.items}
        order = topo_order([it.key for it in decl.items], derived)
        depth = {}
        for key in order:
            parent = derived.get(key) or ""
            depth[key] = depth.get(parent, -1) + 1 if parent else 0
        layers = {}
        for key in order:
            layers.setdefault(depth[key], []).append(key)
        pos = {}
        self.canvas.delete("all")
        for layer in sorted(layers):
            for row, key in enumerate(sorted(layers[layer])):
                x = MARGIN + layer * LAYER_X
                y = MARGIN + row * ROW_Y
                pos[key] = (x, y)
        for key in order:
            parent = derived.get(key) or ""
            if parent and parent in pos:
                x1, y1 = pos[parent]
                x2, y2 = pos[key]
                self.canvas.create_line(x1 + 60, y1, x2 - 60, y2,
                                        arrow=tk.LAST)
        invalidated = {r[2] for r in invalidation if r[1] == "item"}
        for key in order:
            x, y = pos[key]
            color = "#ffd2d2" if key in invalidated else "#d2e5ff"
            self.canvas.create_rectangle(x - 60, y - 18, x + 60, y + 18,
                                         fill=color)
            self.canvas.create_text(x, y, text=key)

        lines = []
        lines.append("账本格式版本: %s (期望 %d)"
                     % (meta.get("format_version", "?"), FORMAT_VERSION))
        lines.append("内容摘要: %s" % (meta.get("content_digest", "?")[:16] + "..."))
        lines.append("已提交批次: %d" % len(batches))
        lines.append("")
        lines.append("最近批次失效集合（稳定拓扑序）:")
        for step, kind, name, reason in invalidation:
            lines.append("  #%d [%s] %s -- %s" % (step, kind, name, reason))
        lines.append("")
        lines.append("逐节点回执来源:")
        for node_id, source, cver, basis in receipts:
            tag = "新回执" if source == "new" else "沿用(v%s)" % basis
            lines.append("  %s: %s, config_version=%s" % (node_id, tag, cver))
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, "\n".join(lines))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="cfgpipe.gui")
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    DiagnosticsApp(args.workdir).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
