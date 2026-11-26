"""Verify the runtime prerequisites described in the task brief."""
from __future__ import annotations

import os
import sys
import tempfile


def main() -> int:
    ok = True
    if sys.version_info < (3, 10):
        print(f"[FAIL] python {sys.version_info[:3]} < 3.10")
        ok = False
    else:
        print(f"[PASS] python {sys.version.split()[0]}")
    for module in ("numpy", "PIL"):
        try:
            imported = __import__(module)
            print(f"[PASS] {module} {getattr(imported, '__version__', '?')}")
        except ImportError:
            print(f"[FAIL] missing {module} (pip install numpy pillow)")
            ok = False
    try:
        import tkinter  # noqa: F401
        print(f"[PASS] tkinter {tkinter.TkVersion}")
    except ImportError:
        print("[FAIL] tkinter missing (apt install python3-tk)")
        ok = False
    with tempfile.NamedTemporaryFile(dir=os.getcwd()) as probe:
        probe.write(b"ok")
    print("[PASS] working directory writable")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
