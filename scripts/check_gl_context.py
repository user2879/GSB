"""First-run OpenGL context check.

Exits 0 and prints a JSON diagnostic when an offscreen context + FBO can be
created; prints an E_GL_CONTEXT structured error and exits 2 otherwise (the
process never crashes with a raw traceback).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcrecon.errors import GLContextError  # noqa: E402
from pcrecon.render_gl import verify_offscreen_context  # noqa: E402


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        info = verify_offscreen_context()
    except GLContextError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True))
        return 2
    print(json.dumps(info, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
