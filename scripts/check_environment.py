#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    try:
        import yaml
    except ImportError:
        print(json.dumps({"ok": False, "error": "PyYAML is not installed"}, ensure_ascii=False, indent=2))
        return 1
    supported = sys.version_info >= (3, 10)
    result = {
        "ok": supported,
        "python": sys.version.split()[0],
        "python_310_or_newer": supported,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if supported else 2


if __name__ == "__main__":
    raise SystemExit(main())

