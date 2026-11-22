"""Local environment readiness checks without external services."""

from __future__ import annotations

import importlib
import os
import shutil
import sqlite3
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any


def _module_status(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def environment_report(workspace: str | Path = ".", *, min_free_mb: int = 64) -> dict[str, Any]:
    workspace_path = Path(workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    python_ok = sys.version_info >= (3, 11)
    checks["python"] = {
        "ok": python_ok,
        "required": ">=3.11",
        "version": ".".join(str(part) for part in sys.version_info[:3]),
        "executable": sys.executable,
    }
    checks["sqlite3"] = {
        "ok": True,
        "module": sqlite3.__file__,
        "library_version": sqlite3.sqlite_version,
    }
    tkinter_status = _module_status("tkinter")
    tkinter_status["required_for_gui"] = True
    tkinter_status["ok"] = tkinter_status["available"]
    checks["tkinter"] = tkinter_status
    numpy_status = _module_status("numpy")
    numpy_status["required"] = False
    numpy_status["note"] = "runtime uses only the Python standard library"
    checks["numpy"] = numpy_status
    writable: dict[str, Any] = {}
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".snapdiag-doctor-",
            dir=workspace_path,
            delete=False,
        ) as handle:
            handle.write(b"ok")
            temporary = Path(handle.name)
        temporary.unlink()
        writable["workspace"] = True
    except OSError as exc:
        writable["workspace"] = False
        writable["workspace_error"] = str(exc)
    for name in ("tmp", "backups"):
        path = workspace_path / name
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-probe"
            probe.write_bytes(b"ok")
            probe.unlink()
            writable[name] = True
        except OSError as exc:
            writable[name] = False
            writable[f"{name}_error"] = str(exc)
    checks["write_access"] = {
        "ok": all(value for key, value in writable.items() if isinstance(value, bool)),
        **writable,
    }
    usage = shutil.disk_usage(workspace_path)
    free_mb = usage.free // (1024 * 1024)
    checks["disk_space"] = {
        "ok": free_mb >= min_free_mb,
        "free_mb": free_mb,
        "minimum_mb": min_free_mb,
    }
    checks["temp_dir"] = {
        "path": tempfile.gettempdir(),
        "writable": os.access(tempfile.gettempdir(), os.W_OK),
    }
    checks["cells"] = {
        "python_stdlib": sysconfig.get_path("stdlib"),
        "platform": sys.platform,
        "container_required": False,
        "network_required": False,
    }
    required_ok = all(
        checks[name]["ok"]
        for name in ("python", "sqlite3", "write_access", "disk_space")
    )
    return {
        "ok": required_ok,
        "operation": "doctor",
        "workspace": str(workspace_path),
        "checks": checks,
        "warnings": [
            "tkinter unavailable: CLI and tests remain usable, GUI cannot launch"
        ]
        if not checks["tkinter"]["ok"]
        else [],
        "offline_numpy_install_example": (
            "python -m pip install --no-index --find-links .\\wheels numpy"
        ),
    }

