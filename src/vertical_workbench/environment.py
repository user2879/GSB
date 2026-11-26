from __future__ import annotations

import multiprocessing as mp
import socket
import sys
from typing import Any


def _mp_ping(queue: "mp.Queue[str]") -> None:
    queue.put("pong")


def check_loopback_tcp() -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                conn, _addr = server.accept()
                with conn:
                    payload = b"loopback-ok"
                    client.sendall(payload)
                    received = conn.recv(1024)
            if received != payload:
                return {"ok": False, "detail": "loopback echo mismatch"}
        return {"ok": True, "host": "127.0.0.1", "port": port}
    except OSError as exc:
        return {"ok": False, "detail": repr(exc)}


def check_multiprocessing() -> dict[str, Any]:
    try:
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        queue = context.Queue()
        process = context.Process(target=_mp_ping, args=(queue,))
        process.start()
        value = queue.get(timeout=5)
        process.join(timeout=5)
        if process.exitcode != 0 or value != "pong":
            return {"ok": False, "exitcode": process.exitcode, "value": value}
        return {"ok": True, "start_method": context.get_start_method()}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"ok": False, "detail": repr(exc)}


def check_numpy() -> dict[str, Any]:
    try:
        import numpy as np

        return {"ok": True, "version": np.__version__}
    except Exception as exc:
        return {"ok": False, "detail": repr(exc)}


def check_tk() -> dict[str, Any]:
    try:
        import tkinter
    except Exception as exc:
        try:
            from .gui import _bootstrap_tkinter

            tkinter, _ttk = _bootstrap_tkinter()
        except Exception:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    try:
        root = tkinter.Tk()
        version = root.tk.call("info", "patchlevel")
        root.destroy()
        return {"ok": True, "tk_version": str(version)}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def check_environment() -> dict[str, Any]:
    return {
        "python": {"ok": sys.version_info >= (3, 10), "version": sys.version.split()[0], "executable": sys.executable},
        "numpy": check_numpy(),
        "tkinter": check_tk(),
        "loopback_tcp": check_loopback_tcp(),
        "multiprocessing": check_multiprocessing(),
    }
