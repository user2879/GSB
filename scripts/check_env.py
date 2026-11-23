#!/usr/bin/env python3
"""Check the offline runtime prerequisites without touching the network."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy
import yaml


def main() -> int:
    ok = sys.version_info >= (3, 11)
    result = {
        "status": "ok" if ok else "failed",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "requires_python": ">=3.11",
        "numpy": numpy.__version__,
        "PyYAML": yaml.__version__,
        "network_required": False,
        "cwd": str(Path.cwd()),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
