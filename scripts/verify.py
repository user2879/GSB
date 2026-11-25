from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scale_review.acceptance import run_acceptance
from scale_review.canonical import write_json_atomic


def main() -> int:
    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )
    unit_output = unit.stdout + "\n" + unit.stderr
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "unittest-output.txt").write_text(unit_output, encoding="utf-8")

    acceptance = run_acceptance(ROOT)
    report = {
        "ok": unit.returncode == 0 and acceptance["ok"],
        "unittest_returncode": unit.returncode,
        "unittest_tail": "\n".join(unit_output.splitlines()[-12:]),
        "acceptance": {
            "ok": acceptance["ok"],
            "checks": acceptance["checks"],
            "report_path": acceptance["report_path"],
        },
    }
    write_json_atomic(ROOT / "artifacts" / "verification-report.json", report)

    lines = [
        "Scale review verification",
        "=========================",
        f"overall: {'PASS' if report['ok'] else 'FAIL'}",
        f"unittest return code: {unit.returncode}",
        report["unittest_tail"],
        "",
        "acceptance checks:",
    ]
    for name, ok in acceptance["checks"].items():
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}")
    lines.extend(
        [
            "",
            f"machine report: {ROOT / 'artifacts' / 'verification-report.json'}",
            f"acceptance evidence: {ROOT / 'artifacts' / 'acceptance-report.json'}",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (ROOT / "artifacts" / "verification-summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
