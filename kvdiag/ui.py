"""Tkinter diagnostic UI for kvdiag (Windows local environment).

The tkinter import is deferred to run_ui so the package remains importable
on headless machines. The UI shows shard declarations, committed index
entries, lease states, convergence conclusions and the structured log, and
can trigger pipeline runs, replay verification and recovery on worker
threads so the window never blocks.
"""

import json
import threading
from pathlib import Path


def run_ui(spec_path, db_path, log_path):
    import tkinter as tk
    from tkinter import messagebox, ttk

    from .errors import KvDiagError
    from .ledger import Ledger
    from .log import StructuredLogger
    from .pipeline import replay_verify, run_pipeline
    from .recovery import recover, write_session_snapshot
    from .spec import load_spec

    root = tk.Tk()
    root.title("kvdiag - index shard & lease expiry diagnostics")
    root.geometry("980x640")

    logger = StructuredLogger(log_path)
    state = {"spec_path": spec_path, "db_path": db_path}

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    def make_tree(parent, columns):
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    shards_tree = make_tree(notebook, ("shard_id", "range", "ttl_ms",
                                       "renew_interval_ms", "max_inflight",
                                       "tolerance"))
    notebook.add(shards_tree.master, text="分片声明")
    entries_tree = make_tree(notebook, ("shard_id", "rule", "key_digest",
                                        "position", "index_version", "batch_id"))
    notebook.add(entries_tree.master, text="索引条目")
    leases_tree = make_tree(notebook, ("lease_id", "shard_id", "version",
                                       "state", "expiry_mono_ms"))
    notebook.add(leases_tree.master, text="租约")
    concl_tree = make_tree(notebook, ("shard_id", "status", "index_pos",
                                      "lease_expiry_pos", "diff", "basis"))
    notebook.add(concl_tree.master, text="收敛结论")

    log_frame = ttk.Frame(notebook)
    notebook.add(log_frame, text="结构化日志")
    log_text = tk.Text(log_frame, wrap=tk.NONE)
    log_text.pack(fill=tk.BOTH, expand=True)

    status_var = tk.StringVar(value="就绪")

    def set_status(text):
        root.after(0, status_var.set, text)

    def fill(tree, rows):
        def do():
            tree.delete(*tree.get_children())
            for row in rows:
                tree.insert("", tk.END, values=row)
        root.after(0, do)

    def refresh():
        try:
            spec = load_spec(state["spec_path"])
            fill(shards_tree, [
                (s.shard_id,
                 "%d-%d" % (s.lo, s.hi) if s.bucket < 0 else "bucket %d" % s.bucket,
                 s.lease.ttl_ms, s.lease.renew_interval_ms,
                 s.lease.max_inflight_renewals, s.tolerance)
                for s in spec.shards])
        except KvDiagError as err:
            set_status("声明加载失败: %s" % err)
        if Path(state["db_path"]).exists():
            ledger = Ledger(state["db_path"])
            fill(entries_tree, [tuple(r[c] for c in ("shard_id", "rule",
                                                     "key_digest", "position",
                                                     "index_version", "batch_id"))
                                for r in ledger.query(
                                    "SELECT * FROM index_entries"
                                    " ORDER BY shard_id, rule, position")])
            fill(leases_tree, [tuple(r[c] for c in ("lease_id", "shard_id",
                                                    "version", "state",
                                                    "expiry_mono_ms"))
                               for r in ledger.query(
                                   "SELECT * FROM leases ORDER BY lease_id")])
            fill(concl_tree, [tuple(r[c] for c in ("shard_id", "status",
                                                   "index_pos",
                                                   "lease_expiry_pos", "diff",
                                                   "basis"))
                              for r in ledger.query(
                                  "SELECT * FROM conclusions ORDER BY shard_id")])
            ledger.close()
        if Path(log_path).exists():
            lines = Path(log_path).read_text(encoding="utf-8").splitlines()
            def show_log():
                log_text.delete("1.0", tk.END)
                log_text.insert(tk.END, "\n".join(lines[-500:]))
            root.after(0, show_log)
        set_status("已刷新 %s" % state["db_path"])

    def background(label, fn):
        def worker():
            set_status("%s 运行中..." % label)
            try:
                result = fn()
                set_status("%s 完成: %s" % (label, json.dumps(result,
                                                              ensure_ascii=False)
                                            [:200]))
            except KvDiagError as err:
                set_status("%s 失败: %s %s" % (label, err.code, err))
                root.after(0, messagebox.showerror, err.code, str(err))
            refresh()
        threading.Thread(target=worker, daemon=True).start()

    def do_run():
        background("流水线", lambda: run_pipeline(
            state["spec_path"], state["db_path"],
            renewals_path=str(Path(state["spec_path"]).parent / "renewals.json"),
            session_out="artifacts/session_snapshot.json", logger=logger))

    def do_replay():
        def fn():
            spec = load_spec(state["spec_path"])
            ledger = Ledger(state["db_path"])
            ok = replay_verify(ledger, spec)
            ledger.close()
            return {"replay_ok": ok}
        background("回放校验", fn)

    def do_recover():
        def fn():
            spec = load_spec(state["spec_path"])
            ledger = Ledger(state["db_path"])
            report = recover(spec, ledger, "artifacts/session_snapshot.json",
                             logger=logger)
            ledger.close()
            return report
        background("恢复", fn)

    def do_snapshot():
        def fn():
            ledger = Ledger(state["db_path"])
            write_session_snapshot(ledger, "artifacts/session_snapshot.json")
            ledger.close()
            return {"session": "artifacts/session_snapshot.json"}
        background("会话快照", fn)

    toolbar = ttk.Frame(root)
    toolbar.pack(fill=tk.X)
    for label, cmd in (("运行流水线", do_run), ("回放校验", do_replay),
                       ("恢复", do_recover), ("写出会话快照", do_snapshot),
                       ("刷新", refresh)):
        ttk.Button(toolbar, text=label, command=cmd).pack(side=tk.LEFT, padx=4,
                                                          pady=4)
    ttk.Label(root, textvariable=status_var, anchor=tk.W).pack(fill=tk.X)

    refresh()
    root.mainloop()
