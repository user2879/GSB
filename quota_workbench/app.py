"""Tkinter 领域数据编辑与操作界面：配额仲裁工作台。"""

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .arbitrate import RULE_VERSION
from .errors import WorkbenchError
from .events import load_events
from .loader import load_declaration
from .ledger import Ledger
from .pipeline import (all_rows, build_view, load_rules, run_to_ledger,
                       scheme_meta)


class WorkbenchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("存储配额仲裁工作台")
        self.geometry("1080x640")
        self.declaration_path = tk.StringVar(value="samples/declaration.json")
        self.events_path = tk.StringVar(value="samples/events.ndjson")
        self.rules_path = tk.StringVar(value="samples/rules.json")
        self.ledger_path = tk.StringVar(value="samples/ledger.sqlite")
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        for label, var in (("声明", self.declaration_path),
                           ("事件流", self.events_path),
                           ("规则", self.rules_path),
                           ("账本", self.ledger_path)):
            ttk.Label(top, text=label).pack(side=tk.LEFT)
            ttk.Entry(top, textvariable=var, width=28).pack(side=tk.LEFT, padx=2)
            ttk.Button(top, text="…", width=2,
                       command=lambda v=var: self._pick(v)).pack(side=tk.LEFT)
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="运行流水线", command=self.run_pipeline).pack(side=tk.LEFT)
        ttk.Button(bar, text="重放校验", command=self.replay).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="查看账本", command=self.show_ledger).pack(side=tk.LEFT)
        self.status = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.status).pack(side=tk.RIGHT)
        cols = ("domain", "volume", "metric", "payload")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        for c, w in zip(cols, (100, 100, 180, 660)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

    def _pick(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _fill(self, rows):
        self.tree.delete(*self.tree.get_children())
        for domain, vid, payload in rows:
            self.tree.insert("", tk.END, values=(
                domain, vid, payload.get("metric", "-"),
                json.dumps(payload, ensure_ascii=False, sort_keys=True)))

    def run_pipeline(self):
        try:
            ledger = Ledger(self.ledger_path.get())
            view, meta = run_to_ledger(
                self.declaration_path.get(), self.events_path.get(),
                self.rules_path.get(), ledger)
            self._fill(all_rows(view))
            self.status.set(f"完成：方案版本 {meta['scheme_version']}，"
                            f"账本摘要 {ledger.content_digest()[:16]}…")
            ledger.close()
        except WorkbenchError as exc:
            messagebox.showerror(exc.code, json.dumps(exc.to_dict(),
                                                      ensure_ascii=False, indent=2))
            self.status.set(f"拒绝：{exc.code}")

    def replay(self):
        try:
            declaration = load_declaration(self.declaration_path.get())
            events = load_events(self.events_path.get())
            rules = load_rules(self.rules_path.get())
            rule_version = rules.get("rule_version", RULE_VERSION)
            view = build_view(declaration, events, rules["target_profile"],
                              rule_version)
            ledger = Ledger(self.ledger_path.get())
            ledger.check_snapshot(
                scheme_meta(declaration, rule_version, 1)["sources_digest"],
                rule_version)
            ledger.replay_verify(all_rows(view))
            ledger.close()
            self.status.set("重放校验通过")
        except WorkbenchError as exc:
            messagebox.showerror(exc.code, json.dumps(exc.to_dict(),
                                                      ensure_ascii=False, indent=2))
            self.status.set(f"重放失败：{exc.code}")

    def show_ledger(self):
        ledger = Ledger(self.ledger_path.get())
        rows = [(d, v, json.loads(p)) for d, v, _, p in ledger.all_rows()]
        ledger.close()
        self._fill(rows)
        self.status.set(f"账本行数 {len(rows)}")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    WorkbenchApp().mainloop()


if __name__ == "__main__":
    main()
