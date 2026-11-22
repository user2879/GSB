"""Read-only interpreter, standard-library, dependency, and path self-check."""

from __future__ import annotations

import importlib
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .config import Declarations


def run_selfcheck(
    declarations: Declarations | None = None,
    *,
    require_tkinter: bool = False,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    version = sys.version_info
    python_ok = version >= (3, 10)
    checks["python"] = {
        "ok": python_ok,
        "required": ">=3.10",
        "actual": f"{version.major}.{version.minor}.{version.micro}",
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }
    sqlite_ok = sqlite3.sqlite_version_info >= (3, 0, 0)
    checks["sqlite3"] = {
        "ok": sqlite_ok,
        "required": "std sqlite3",
        "actual": sqlite3.sqlite_version,
    }
    try:
        numpy = importlib.import_module("numpy")
        checks["numpy"] = {"ok": True, "required": "installed", "actual": numpy.__version__}
    except Exception as exc:
        checks["numpy"] = {"ok": False, "required": "installed", "error": f"{type(exc).__name__}: {exc}"}
    try:
        tkinter = importlib.import_module("tkinter")
        checks["tkinter"] = {"ok": True, "required": "std tkinter", "actual": str(tkinter.TkVersion)}
    except Exception as exc:
        checks["tkinter"] = {"ok": False, "required": "std tkinter", "error": f"{type(exc).__name__}: {exc}"}

    if declarations is not None:
        paths = {
            key: {
                "path": str(path),
                "exists": Path(path).exists(),
                "parent_exists": Path(path).parent.exists(),
            }
            for key, path in declarations.paths.items()
        }
        checks["declarations"] = {
            "ok": True,
            "parameter_fingerprint": declarations.parameter_fingerprint(),
            "format_versions": list(declarations.format_versions),
            "topic_count": len(declarations.topics),
            "consumer_group_count": len(declarations.groups),
            "paths": paths,
        }

    required_ok = checks["python"]["ok"] and checks["sqlite3"]["ok"] and checks["numpy"]["ok"]
    tkinter_ok = checks["tkinter"]["ok"]
    if require_tkinter:
        required_ok = required_ok and tkinter_ok
    if not required_ok:
        status = "fail"
    elif not tkinter_ok:
        status = "pass_with_warnings"
    else:
        status = "pass"
    warnings = []
    if not tkinter_ok:
        warnings.append(
            "tkinter import failed; core CLI remains usable, but the Windows GUI requirement is not satisfied"
        )
    return {
        "status": status,
        "read_only": True,
        "checks": checks,
        "warnings": warnings,
    }

