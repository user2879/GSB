from __future__ import annotations

import ctypes
import glob
import importlib
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_tk_from_local_root() -> str | None:
    candidates = []
    if os.environ.get("TK_ROOT"):
        candidates.append(Path(os.environ["TK_ROOT"]))
    candidates.extend(
        [
            PROJECT_ROOT / ".tk-runtime",
            Path.home() / ".local" / "tkroot",
            Path("/home/zsl/.local/tkroot"),
        ]
    )
    root = next((path for path in candidates if path.exists()), None)
    if root is None:
        return None
    root = root.resolve()
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    python_lib = root / "usr" / "lib" / version
    dynload = python_lib / "lib-dynload"
    for path in (str(python_lib), str(dynload)):
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)
    for library_dir in (
        root / "usr" / "lib" / "x86_64-linux-gnu",
        root / "usr" / "lib",
    ):
        for library in sorted(glob.glob(str(library_dir / "*.so*"))):
            try:
                ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    tcl = root / "usr" / "share" / "tcltk" / "tcl8.6"
    tk = root / "usr" / "share" / "tcltk" / "tk8.6"
    if tcl.exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl))
    if tk.exists():
        os.environ.setdefault("TK_LIBRARY", str(tk))
    return str(root)


def check_tkinter() -> dict[str, Any]:
    bootstrapped = None
    try:
        import tkinter
    except ImportError:
        bootstrapped = bootstrap_tk_from_local_root()
        try:
            import tkinter
        except Exception as exc:  # environment diagnostic should not raise
            return {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "bootstrap_root": bootstrapped,
            }
    try:
        import tkinter

        return {
            "available": True,
            "tk_version": str(tkinter.TkVersion),
            "tcl_version": str(tkinter.TclVersion),
            "module": tkinter.__file__,
            "bootstrap_root": bootstrapped,
            "display": os.environ.get("DISPLAY"),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pytest"):
        try:
            module = importlib.import_module(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError as exc:
            packages[name] = {"available": False, "error": str(exc)}
    return {
        "ok": True,
        "python": sys.version,
        "executable": sys.executable,
        "sqlite3": {"available": True, "version": sqlite3.sqlite_version},
        "tkinter": check_tkinter(),
        "packages": packages,
    }
