from __future__ import annotations

import ctypes
import os
from pathlib import Path


def import_tkinter():
    try:
        import tkinter as tk

        return tk
    except ImportError:
        root = Path(__file__).resolve().parents[1] / "vendor" / "tk" / "usr"
        lib_dir = root / "lib" / "x86_64-linux-gnu"
        tcl = lib_dir / "libtcl8.6.so"
        tk_lib = lib_dir / "libtk8.6.so"
        if not tcl.exists() or not tk_lib.exists():
            raise
        ctypes.CDLL(str(tcl), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(tk_lib), mode=ctypes.RTLD_GLOBAL)
        os.environ.setdefault("TCL_LIBRARY", str(root / "share" / "tcltk" / "tcl8.6"))
        os.environ.setdefault("TK_LIBRARY", str(root / "share" / "tcltk" / "tk8.6"))
        import tkinter as tk

        return tk
