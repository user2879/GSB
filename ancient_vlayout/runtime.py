"""Runtime import and package validation."""

from __future__ import annotations

import importlib
from typing import Any

from .errors import E_DEPENDENCY_VERSION_MISMATCH, E_RUNTIME_INVALID


REQUIRED_PACKAGES = {
    "numpy": "1.26.4",
    "PIL": "10.3.0",
    "pytest": "8.2.0",
}


def validate_runtime(include_test_packages: bool = True) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    packages: dict[str, str] = {}
    required = dict(REQUIRED_PACKAGES)
    if not include_test_packages:
        required.pop("pytest", None)
    for name, expected in required.items():
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - environment failure path
            diagnostics.append(
                {
                    "code": E_RUNTIME_INVALID,
                    "message": f"required package {name} failed to import: {exc}",
                    "package": name,
                }
            )
            continue
        actual = str(getattr(module, "__version__", "unknown"))
        packages[name] = actual
        if actual != expected:
            diagnostics.append(
                {
                    "code": E_DEPENDENCY_VERSION_MISMATCH,
                    "message": f"{name} version {actual} differs from pinned {expected}",
                    "package": name,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return {
        "ok": not any(item["code"] == E_RUNTIME_INVALID for item in diagnostics),
        "packages": packages,
        "pinned": required,
        "diagnostics": diagnostics,
    }
