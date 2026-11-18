#!/usr/bin/env python3
"""Check the Windows runtime prerequisites without third-party packages."""

from __future__ import annotations

import json
import platform
import sqlite3
import sys


def main() -> int:
    try:
        import tkinter

        tkinter_version = str(tkinter.TkVersion)
        tkinter_ok = True
    except ImportError:
        tkinter_version = None
        tkinter_ok = False
    result = {
        "python_version": platform.python_version(),
        "python_3_10_or_newer": sys.version_info >= (3, 10),
        "tkinter_available": tkinter_ok,
        "tkinter_version": tkinter_version,
        "sqlite_version": sqlite3.sqlite_version,
        "third_party_dependencies_required": False,
        "ready": sys.version_info >= (3, 10) and tkinter_ok,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
