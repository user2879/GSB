"""Optional Tk control panel.

The service, tests and ledger tools never import tkinter.  This keeps the core
usable with embedded Python distributions that omit tcl/tk while still
satisfying the requested optional desktop console.
"""

from __future__ import annotations

import json
from typing import Any

from .engine import Engine, EngineConfig
from .logging import StructuredLogger
from .server import RunningServer, request


def launch(db_path: str, log_path: str) -> int:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Transaction Message Middleware - Local Loopback")
    root.geometry("920x640")
    state: dict[str, Any] = {"running": None, "engine": None, "address": None, "thread": None}

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="Database").pack(side="left")
    db_var = tk.StringVar(value=db_path)
    ttk.Entry(top, textvariable=db_var, width=70).pack(side="left", padx=4)
    port_var = tk.StringVar(value="0")
    ttk.Label(top, text="Port").pack(side="left")
    ttk.Entry(top, textvariable=port_var, width=8).pack(side="left", padx=4)
    status_var = tk.StringVar(value="stopped")
    ttk.Label(top, textvariable=status_var).pack(side="left", padx=12)

    body = ttk.Frame(root, padding=(8, 0, 8, 8))
    body.pack(fill="both", expand=True)
    output = tk.Text(body, wrap="none", font=("Consolas", 10))
    output.pack(fill="both", expand=True)

    def append(value: Any) -> None:
        output.insert("end", json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        output.see("end")

    def start() -> None:
        if state["running"] is not None:
            return
        logger = StructuredLogger(log_path)
        engine = Engine(db_var.get(), config=EngineConfig(), logger=logger)
        running = RunningServer(engine, port=int(port_var.get() or 0))
        running.start()
        state.update(running=running, engine=engine, address=running.address)
        host, port = running.address
        status_var.set(f"listening {host}:{port}")
        append({"started": True, "host": host, "port": port, "recovery": engine.recovery_report})

    def stop() -> None:
        running = state.get("running")
        if running is None:
            return
        report = running.stop(report=True)
        status_var.set("stopped")
        append({"stopped": True, "exit_report": report})
        state.update(running=None, engine=None, address=None)

    def refresh() -> None:
        address = state.get("address")
        if not address:
            append({"status": "not running"})
            return
        try:
            append({"inventory": request(*address, {"op": "inventory"})})
        except Exception as exc:
            append({"error": repr(exc)})

    def verify() -> None:
        engine = state.get("engine")
        if engine is None:
            append({"error": "server is not running"})
            return
        from .verify import verify_database

        append({"verify": verify_database(engine.store)})

    controls = ttk.Frame(root, padding=(8, 0, 8, 8))
    controls.pack(fill="x")
    ttk.Button(controls, text="Start", command=start).pack(side="left")
    ttk.Button(controls, text="Stop", command=stop).pack(side="left", padx=4)
    ttk.Button(controls, text="Refresh", command=refresh).pack(side="left", padx=4)
    ttk.Button(controls, text="Verify", command=verify).pack(side="left", padx=4)
    ttk.Label(
        controls,
        text="Loopback only. All query/pull timing is measured from real socket and SQLite work.",
    ).pack(side="left", padx=12)

    def on_close() -> None:
        stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    append(
        {
            "hint": "Start the server, then use main.py send/scenario or tools/query_stub.py.",
            "db": db_path,
            "log": log_path,
        }
    )
    root.mainloop()
    return 0
