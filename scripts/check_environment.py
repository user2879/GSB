#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sqlite3
import sys


def main() -> int:
    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "sqlite3": sqlite3.sqlite_version,
        "python_ok": sys.version_info >= (3, 11),
    }
    try:
        import tkinter

        report["tkinter"] = tkinter.TkVersion
    except Exception as exc:
        report["tkinter"] = None
        report["tkinter_note"] = f"optional GUI unavailable: {type(exc).__name__}: {exc}"
    try:
        import numpy

        report["numpy"] = numpy.__version__
    except Exception as exc:
        report["numpy"] = None
        report["numpy_note"] = (
            "optional only; the deterministic fixed-point core uses Python integers and does not require numpy"
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["python_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
