from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def verify_write_permission(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".write-check-", dir=target, delete=False) as handle:
        probe = Path(handle.name)
        handle.write(b"chaindiag-write-check\n")
    try:
        with probe.open("r", encoding="utf-8") as stream:
            ok = stream.read().strip() == "chaindiag-write-check"
        if not ok:
            raise OSError("write probe content mismatch")
        return {"path": str(target.resolve()), "writable": True, "access_w_ok": os.access(target, os.W_OK)}
    finally:
        probe.unlink(missing_ok=True)
