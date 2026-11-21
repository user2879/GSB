#!/usr/bin/env python3
"""Check the local Windows prerequisites without network access."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-tk", action="store_true")
    parser.add_argument("--open-window", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_ok": True,
    }
    try:
        import numpy

        report["numpy_version"] = numpy.__version__
        report["numpy_ok"] = True
    except Exception as error:  # pragma: no cover - environment dependent
        report["numpy_version"] = None
        report["numpy_ok"] = False
        report["numpy_error"] = str(error)

    tkinter_available = False
    try:
        import tkinter

        report["tkinter_version"] = tkinter.TkVersion
        tkinter_available = True
        if args.open_window:
            root = tkinter.Tk()
            root.title("mlpipe-workbench environment check")
            root.geometry("420x140")
            tkinter.Label(root, text="Tkinter is available. Closing in 2 seconds.").pack(pady=36)
            root.after(2000, root.destroy)
            root.mainloop()
    except Exception as error:  # pragma: no cover - environment dependent
        report["tkinter_version"] = None
        report["tkinter_error"] = str(error)
    report["tkinter_ok"] = tkinter_available

    try:
        with tempfile.NamedTemporaryFile(dir=Path.cwd(), prefix=".write-test-", delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
        report["working_directory_writable"] = True
    except Exception as error:  # pragma: no cover - environment dependent
        report["working_directory_writable"] = False
        report["write_error"] = str(error)

    required_ok = bool(
        report["python_ok"]
        and report["sqlite_ok"]
        and report["numpy_ok"]
        and report["working_directory_writable"]
        and (tkinter_available or not args.require_tk)
    )
    report["status"] = "OK" if required_ok else "ERROR"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if required_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
