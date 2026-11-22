"""Minimal Tkinter front-end (optional). Tkinter ships with the standard
Windows python.org installer; on platforms without Tk the GUI is simply
unavailable and the CLI remains fully functional.
"""

import os
import threading
import traceback


def run_gui(base_dir="."):
    import tkinter as tk
    from tkinter import filedialog

    from . import cli  # reuse CLI command implementations

    root = tk.Tk()
    root.title("colval - columnar vectorized execution verifier")
    base_var = tk.StringVar(value=os.path.abspath(base_dir))

    top = tk.Frame(root)
    top.pack(fill="x", padx=6, pady=6)
    tk.Label(top, text="workspace:").pack(side="left")
    tk.Entry(top, textvariable=base_var, width=48).pack(side="left", padx=4)

    log = tk.Text(root, width=100, height=32)
    log.pack(fill="both", expand=True, padx=6, pady=6)

    def append(text):
        log.insert("end", text + "\n")
        log.see("end")

    def run_action(fn):
        def worker():
            import contextlib
            import io
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    fn()
                append(buf.getvalue())
            except Exception:
                append(traceback.format_exc())
        threading.Thread(target=worker, daemon=True).start()

    def pick_dir():
        d = filedialog.askdirectory(initialdir=base_var.get())
        if d:
            base_var.set(d)

    class Args:
        pass

    def mk(name):
        a = Args()
        a.base = base_var.get()
        if name == "run":
            a.crash_after = None
        return a

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=6, pady=4)
    tk.Button(btns, text="browse", command=pick_dir).pack(side="left")
    tk.Button(btns, text="init-sample",
              command=lambda: run_action(
                  lambda: cli._cmd_init_sample(mk("init-sample")))
              ).pack(side="left", padx=4)
    tk.Button(btns, text="run",
              command=lambda: run_action(lambda: cli._cmd_run(mk("run")))
              ).pack(side="left", padx=4)
    tk.Button(btns, text="verify",
              command=lambda: run_action(lambda: cli._cmd_verify(mk("verify")))
              ).pack(side="left", padx=4)

    root.mainloop()
