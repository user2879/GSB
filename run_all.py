"""Single entry point used by acceptance runs on Windows or any OS.

Steps: prepare deterministic samples -> unit tests -> demo scenarios ->
fault injection smoke checks -> consistency command.  All artifacts stay
inside the project directory (``colstore/data`` and ``colstore/sample_data``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from colstore.cli import main as cli_main  # noqa: E402
from colstore.demo import run_demo  # noqa: E402
from colstore.prepare_samples import prepare  # noqa: E402


def main() -> int:
    print("[1/5] prepare deterministic samples")
    prepare()

    print("[2/5] unit tests")
    test_run = subprocess.run(
        [sys.executable, "-m", "unittest", "colstore.tests.test_core", "-v"],
        cwd=ROOT,
    )
    if test_run.returncode != 0:
        return test_run.returncode

    print("[3/5] demo scenarios")
    demo_summary = run_demo()
    print("demo verdict:", demo_summary["verdict"])

    print("[4/5] fault injection business codes")
    for kind in ("payload_truncation", "encoding_mismatch", "zero_budget"):
        code = cli_main(["inject", "--kind", kind])
        print(f"  {kind} -> exit {code} (expected 2)")
        assert code == 2

    print("[5/5] cross-parallelism / cross-encoding consistency")
    code = cli_main(["check", "--query", "query_and.json", "--parallelism", "4"])
    assert code == 0

    report_path = os.path.join(ROOT, "colstore", "data", "run_all_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "demo_verdict": demo_summary["verdict"],
                "checks_total": len(demo_summary["checks"]),
                "fault_codes": ["E_PAYLOAD", "E_ENCODING", "E_BUDGET"],
                "consistency": "PASS",
            },
            handle,
            indent=2,
        )
    print("report written:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
