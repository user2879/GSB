from __future__ import annotations

import json
import sys
from pathlib import Path

from workbench.demo import run_demo


def main() -> int:
    report = run_demo(Path(__file__).resolve().parent)
    summary = {
        "ok": report["ok"],
        "report": "artifacts/reports/verification-report.json",
        "consistency": report["consistency"]["equal"],
        "injection_consistent": report["injection_diagnostic"]["consistent"],
        "all_injections_hit": report["injection_diagnostic"]["all_injections_hit"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
