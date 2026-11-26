"""Best-effort Tk bootstrap for WSL images missing python3-tk runtime files."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _enable_local_tk_libraries() -> list[str]:
    loaded: list[str] = []
    library_root = PROJECT_ROOT / ".local-tk/usr/lib/x86_64-linux-gnu"
    share_root = PROJECT_ROOT / ".local-tk/usr/share/tcltk"
    python_root = PROJECT_ROOT / ".local-tk/usr/lib/python3.11"
    if library_root.exists():
        for library_name in ("libXft.so.2", "libXss.so.1", "libtcl8.6.so", "libtk8.6.so"):
            candidate = library_root / library_name
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(str(candidate))
                except OSError:
                    pass
        tcl_library = share_root / "tcl8.6"
        tk_library = share_root / "tk8.6"
        if tcl_library.exists():
            os.environ.setdefault("TCL_LIBRARY", str(tcl_library))
        if tk_library.exists():
            os.environ.setdefault("TK_LIBRARY", str(tk_library))
    feedback_site_packages = Path("/home/zsl/.feedback/python-base/lib/python3.11/site-packages")
    tkinter_init = feedback_site_packages / "tkinter/__init__.py"
    if tkinter_init.exists() and str(feedback_site_packages) not in sys.path:
        sys.path.append(str(feedback_site_packages))
    elif python_root.exists():
        for path in (python_root, python_root / "lib-dynload"):
            if path.exists() and str(path) not in sys.path:
                sys.path.append(str(path))
    return loaded


def import_tkinter():
    try:
        import tkinter  # noqa: F401

        return tkinter, None
    except Exception as first_error:  # noqa: BLE001
        _enable_local_tk_libraries()
        try:
            import tkinter  # type: ignore  # noqa: F401

            return tkinter, None
        except Exception as second_error:  # noqa: BLE001
            return None, f"{type(first_error).__name__}: {first_error}; retry: {type(second_error).__name__}: {second_error}"


def tk_status(create_root: bool = False) -> dict[str, object]:
    tkinter, error = import_tkinter()
    if tkinter is None:
        return {"tkinter_available": False, "error": error}
    status = {"tkinter_available": True, "tk_version": str(tkinter.TkVersion)}
    if create_root:
        try:
            root = tkinter.Tk()
            status["display"] = os.environ.get("DISPLAY", "")
            status["window_created"] = True
            status["patchlevel"] = str(root.tk.call("info", "patchlevel"))
            root.destroy()
        except Exception as exc:  # noqa: BLE001
            status["window_created"] = False
            status["window_error"] = f"{type(exc).__name__}: {exc}"
    return status
