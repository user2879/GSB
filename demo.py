#!/usr/bin/env python3
import json
from pathlib import Path

from candidate_diagnostics.demo import run_demo


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    report = run_demo(root)
    print(json.dumps({
        "report": str(root / "artifacts" / "diagnostic-report.json"),
        "markdown": str(root / "artifacts" / "diagnostic-report.md"),
        "requests": report["sample_request_count"],
        "injections_hit": f"{report['injections']['hit_count']}/{report['injections']['total']}",
        "incremental_matches_blank": report["incremental"]["final_matches_blank_rebuild"],
        "concurrency_matches_serial": report["concurrency"]["business_fields_match_serial"],
    }, ensure_ascii=False, indent=2))
