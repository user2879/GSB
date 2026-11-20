"""Check temp-directory creation and atomic file replacement on this host.

Run this script on Windows before relying on snapshot recovery there:
    python scripts/check_windows_atomic.py
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import tempfile


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="parseapp-atomic-check-") as temp_dir:
        root = Path(temp_dir)
        target = root / "CURRENT.json"
        target.write_text("old", encoding="utf-8")
        candidate = root / "candidate.json"
        candidate.write_text("new", encoding="utf-8")
        replaced = False
        try:
            candidate.replace(target)
            replaced = target.read_text(encoding="utf-8") == "new"
        except OSError as exc:
            result = {
                "platform": platform.system(),
                "temp_dir_created": True,
                "atomic_replace_supported": False,
                "error": str(exc),
            }
        else:
            result = {
                "platform": platform.system(),
                "temp_dir_created": True,
                "atomic_replace_supported": replaced,
                "error": None,
            }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if replaced else 1


if __name__ == "__main__":
    raise SystemExit(main())
