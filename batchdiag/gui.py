"""Optional Tkinter dashboard.

Tk ships with standard Python on Windows/macOS desktop installs. On a
headless Linux host without the Tk shared libraries, importing tkinter
raises ImportError; callers (the CLI ``gui`` command) catch that and fall
back to the deterministic CLI output instead of failing.
"""


def launch(requests_path, workers_path, rules_path, ledger_path):  # pragma: no cover
    import json
    import tkinter as tk
    from tkinter import ttk

    from .ledger import open_ledger
    from .spec import load_spec

    load_spec(requests_path, workers_path, rules_path)
    ledger, status = open_ledger(ledger_path)
    rows = ledger.committed_batches()
    ledger.close()

    root = tk.Tk()
    root.title("batchdiag attribution dashboard")
    columns = ("batch_id", "worker", "seq", "output_digest")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
    for batch_id, result_json, _committed in rows:
        result = json.loads(result_json)
        tree.insert("", "end", values=(
            batch_id, result["worker_id"],
            batch_id.split(":")[2], result["output_digest"]))
    tree.pack(expand=True, fill="both")
    ttk.Label(root, text=f"snapshot: {status.action} "
                         f"v{status.format_version}").pack()
    root.mainloop()
