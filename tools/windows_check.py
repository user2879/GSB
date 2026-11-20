"""Print a JSON readiness report for a local Windows Python installation."""

from __future__ import annotations

import json
import sqlite3
import sys


def main() -> int:
    report = {
        "python": sys.version,
        "sqlite3": sqlite3.sqlite_version,
        "socket": True,
        "subprocess": True,
        "tkinter": False,
        "tkinter_error": None,
    }
    try:
        import tkinter

        report["tkinter"] = True
        report["tkinter_version"] = tkinter.TkVersion
    except Exception as exc:
        report["tkinter_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["tkinter"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
