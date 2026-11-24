#!/usr/bin/env python3
"""Stop workers recorded by start_workers.py."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
runtime_state = ROOT / "artifacts" / "runtime" / "workers.json"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    if not runtime_state.exists():
        print("no worker state file; nothing to stop")
        return 0
    state = json.loads(runtime_state.read_text(encoding="utf-8"))
    failures: list[dict[str, object]] = []
    for record in state.get("workers", []):
        pid = int(record["pid"])
        if not _alive(pid):
            continue
        try:
            if os.name == "nt":
                os.kill(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            failures.append({"pid": pid, "error": str(exc)})
    runtime_state.unlink(missing_ok=True)
    result = {"stopped": True, "failures": failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
