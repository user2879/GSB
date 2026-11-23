"""Optional Windows-friendly tkinter dashboard.

The CLI is the primary testable interface.  This dashboard deliberately shells
out to the CLI so the GUI and command line share exactly the same validation,
ledger and error semantics.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any


def launch_gui(*, config: str | None, requests: str | None, workdir: str) -> dict[str, Any]:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Inference Gateway Incremental Diagnostics")
    root.geometry("980x720")
    result: dict[str, Any] = {"ok": True, "command": "gui", "closed": False}

    config_var = tk.StringVar(value=config or "samples/base.json")
    requests_var = tk.StringVar(value=requests or "samples/requests.ndjson")
    workdir_var = tk.StringVar(value=workdir)
    before_var = tk.StringVar(value="samples/base.json")
    after_var = tk.StringVar(value="samples/update_rule.json")
    spawn_var = tk.BooleanVar(value=True)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)
    rows = [
        ("Config / after", config_var),
        ("Requests NDJSON", requests_var),
        ("Workdir", workdir_var),
        ("Validate before", before_var),
        ("Validate after", after_var),
    ]
    for row, (label, variable) in enumerate(rows):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
    frame.columnconfigure(1, weight=1)

    output = tk.Text(frame, wrap="none", height=30)
    output.grid(row=7, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
    frame.rowconfigure(7, weight=1)
    status = tk.StringVar(value="Ready")
    ttk.Label(frame, textvariable=status).grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))

    def append(text: str) -> None:
        output.insert(tk.END, text + "\n")
        output.see(tk.END)

    def run_async(command: list[str]) -> None:
        status.set("Running: " + " ".join(command[:3]) + " ...")

        def worker() -> None:
            completed = subprocess.run(command, capture_output=True, text=True)
            text = completed.stdout
            if completed.stderr:
                text += ("\n" if text else "") + completed.stderr
            if not text.strip():
                text = json.dumps({"ok": completed.returncode == 0, "exit_code": completed.returncode})
            try:
                parsed = json.loads(text)
                text = json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
            root.after(0, lambda: (append(text), status.set(f"Finished (exit {completed.returncode})")))

        threading.Thread(target=worker, daemon=True).start()

    def base_command(subcommand: str) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "gateway_diag",
            "--pretty",
            subcommand,
            "--requests",
            requests_var.get(),
            "--workdir",
            workdir_var.get(),
        ]
        if spawn_var.get():
            command.append("--spawn-backends")
        return command

    def do_run() -> None:
        command = base_command("run")
        command.extend(["--config", config_var.get()])
        append("$ " + " ".join(command))
        run_async(command)

    def do_validate() -> None:
        command = base_command("validate")
        command.extend(["--before", before_var.get(), "--after", after_var.get()])
        append("$ " + " ".join(command))
        run_async(command)

    def do_verify() -> None:
        command = base_command("verify")
        command.extend(["--config", config_var.get()])
        append("$ " + " ".join(command))
        run_async(command)

    def do_snapshot() -> None:
        command = [
            sys.executable,
            "-m",
            "gateway_diag",
            "--pretty",
            "snapshot",
            "--config",
            config_var.get(),
            "--requests",
            requests_var.get(),
            "--workdir",
            workdir_var.get(),
        ]
        append("$ " + " ".join(command))
        run_async(command)

    button_bar = ttk.Frame(frame)
    button_bar.grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 0))
    ttk.Checkbutton(button_bar, text="Spawn loopback mock", variable=spawn_var).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Button(button_bar, text="Run / Update", command=do_run).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_bar, text="Validate Incremental vs Full", command=do_validate).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_bar, text="Independent Verify", command=do_verify).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_bar, text="Show Snapshot", command=do_snapshot).pack(side=tk.LEFT, padx=3)
    ttk.Button(button_bar, text="Clear Output", command=lambda: output.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=3)

    def on_close() -> None:
        result["closed"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return result
