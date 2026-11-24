from __future__ import annotations

import importlib
import json
import platform
import socket
import sys
from typing import Any


def check_port(host: str = "127.0.0.1", start: int = 20000, end: int = 29999) -> dict[str, Any]:
    for port in range(start, end + 1):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, port))
            return {"ok": True, "host": host, "port": port, "range": [start, end]}
        except OSError:
            continue
        finally:
            probe.close()
    return {"ok": False, "host": host, "range": [start, end], "error": "no free loopback port"}


def main() -> int:
    checks: dict[str, Any] = {
        "python": {
            "ok": sys.version_info >= (3, 11),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "modules": {},
        "numerical": {},
        "loopback": check_port(),
    }
    for name in ("tkinter", "sqlite3", "socket", "subprocess", "json", "threading"):
        try:
            importlib.import_module(name)
            checks["modules"][name] = {"ok": True}
        except Exception as exc:
            checks["modules"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        numpy = importlib.import_module("numpy")
        checks["numerical"] = {"ok": numpy.__version__ == "1.26.4", "numpy": numpy.__version__}
    except Exception as exc:
        checks["numerical"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    critical_ok = (
        checks["python"]["ok"]
        and checks["loopback"]["ok"]
        and all(value["ok"] for value in checks["modules"].values())
        and checks["numerical"]["ok"]
    )
    print(json.dumps({"ok": critical_ok, **checks}, sort_keys=True, indent=2))
    return 0 if critical_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
