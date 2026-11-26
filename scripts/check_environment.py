#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vertical_workbench.environment import check_environment


if __name__ == "__main__":
    report = check_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    ok = report["python"]["ok"] and report["numpy"]["ok"] and report["loopback_tcp"]["ok"] and report["multiprocessing"]["ok"]
    raise SystemExit(0 if ok else 1)
