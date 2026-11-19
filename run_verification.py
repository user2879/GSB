"""Convenience entry point for the deterministic verification suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bone_retarget.verify import run_verification


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bone retarget verification suite.")
    parser.add_argument("--out", default="samples/generated", help="output directory")
    parser.add_argument("--no-clean", action="store_true", help="keep existing output files")
    args = parser.parse_args()
    report = run_verification(Path(args.out), clean=not args.no_clean)
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": report["case_count"],
                "anomaly_count": report["anomaly_count"],
                "error_coverage": report["error_coverage"],
                "manifest_sha256": report["manifest_sha256"],
                "report": str(Path(args.out) / "report.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
