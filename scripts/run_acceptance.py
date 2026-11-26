from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdr.acceptance import run_acceptance


if __name__ == "__main__":
    report = run_acceptance(ROOT)
    summary = {
        "passed": report["passed"],
        "edit_parity": report["edit_parity"]["passed"],
        "precise_invalidation": report["precise_invalidation"]["passed"],
        "concurrency_and_budgets": report["concurrency_and_budgets"]["passed"],
        "recovery": report["recovery"]["passed"],
        "fingerprint": report["main_run"]["fingerprint"],
        "report": "reports/acceptance-report.md",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
