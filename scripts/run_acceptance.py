"""Run the full verification ladder and persist reproducible acceptance artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing_workbench.util import atomic_write_json, atomic_write_text
from routing_workbench.verification import run_verification


def main() -> int:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "acceptance-report.json"
    report = run_verification(
        samples_dir=ROOT / "samples",
        workdir=ROOT / ".verify",
        report_path=report_path,
        run_pytest=True,
    )
    pytest_check = next((check for check in report["checks"] if check["name"] == "pytest_suite"), None)
    if pytest_check is not None:
        output = pytest_check["details"].get("output", [])
        atomic_write_text(artifacts / "pytest.txt", "\n".join(output) + "\n")
    atomic_write_json(artifacts / "acceptance-summary.json", {
        "all_passed": report["all_passed"],
        "checks": [
            {"name": check["name"], "passed": check["passed"]}
            for check in report["checks"]
        ],
        "sample": report["sample"],
        "environment": report["environment"],
    })
    print(json.dumps({
        "all_passed": report["all_passed"],
        "checks": len(report["checks"]),
        "digest": report["sample"]["business_digest"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

