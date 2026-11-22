from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    result = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "numpy": None,
        "tkinter": None,
        "core_ready": sys.version_info >= (3, 10),
    }
    try:
        import numpy

        result["numpy"] = numpy.__version__
    except Exception as exc:
        result["numpy"] = f"unavailable: {exc}"
    try:
        import tkinter

        result["tkinter"] = tkinter.TkVersion
    except Exception as exc:
        result["tkinter"] = f"unavailable: {exc}"
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["core_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
