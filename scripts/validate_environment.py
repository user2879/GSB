from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local slowdiag environment")
    parser.add_argument("--workspace", default="workbench")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks: dict[str, Any] = {}
    checks["python"] = {
        "ok": sys.version_info >= (3, 10),
        "version": sys.version.split()[0],
        "executable": sys.executable,
    }
    checks["sqlite3"] = {"ok": True, "version": sqlite3.sqlite_version}
    try:
        import tkinter

        checks["tkinter"] = {"ok": True, "version": tkinter.TkVersion}
    except Exception as exc:
        checks["tkinter"] = {"ok": False, "error": repr(exc)}
    try:
        import numpy

        checks["numpy"] = {"ok": True, "version": numpy.__version__}
    except Exception as exc:
        checks["numpy"] = {"ok": False, "error": repr(exc)}

    workspace = Path(args.workspace).resolve(strict=False)
    write_ok = False
    write_error = None
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=workspace, delete=True, prefix=".write-check-"
        ) as handle:
            handle.write("ok")
            handle.flush()
        write_ok = True
    except Exception as exc:
        write_error = repr(exc)
    checks["workspace_write"] = {
        "ok": write_ok,
        "workspace": str(workspace),
        **({"error": write_error} if write_error else {}),
    }
    passed = all(item["ok"] for item in checks.values())
    output = {"ok": passed, "checks": checks}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for name, item in checks.items():
            status = "OK" if item["ok"] else "FAIL"
            print(f"{name:16} {status:5} {item}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
